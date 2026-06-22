// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! GroupByJoinToWindow: A logical rewrite for the Athena-style computation reuse pattern
//!
//! This optimizer rule implements the GroupByJoinToWindow equivalence from Amazon's
//! "Computation Reuse via Fusion in Amazon Athena" paper.
//!
//! It rewrites this logical shape:
//!
//! ```text
//! Join(
//!   left = P1,
//!   right = Aggregate {
//!     input = P2,
//!     group_expr = G,
//!     aggr_expr = A
//!   },
//!   on = P1.K = G
//! )
//! ```
//!
//! into:
//!
//! ```text
//! Projection(
//!   WindowAggr(
//!     input = fused_or_reused_P,
//!     window_expr = A OVER (PARTITION BY K)
//!   )
//! )
//! ```
//!
//! This avoids duplicated computation of the aggregate when P1 and P2 are the same
//! logical input by using a window aggregate instead.

use crate::optimizer::ApplyOrder;
use crate::{OptimizerConfig, OptimizerRule};
use datafusion_common::tree_node::Transformed;
use datafusion_common::{DFSchema, Result};
use datafusion_expr::expr::AggregateFunction;
use datafusion_expr::logical_plan::{Aggregate, Join, LogicalPlan, Projection};
use datafusion_expr::{col, Expr, JoinType, WindowFrame};
use std::sync::Arc;

/// Optimizer rule that rewrites eligible group-by-join patterns into window aggregates.
#[derive(Debug, Default)]
pub struct GroupByJoinToWindow;

impl GroupByJoinToWindow {
    #[expect(missing_docs)]
    pub fn new() -> Self {
        Self {}
    }
}

impl OptimizerRule for GroupByJoinToWindow {
    fn name(&self) -> &str {
        "group_by_join_to_window"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::BottomUp)
    }

    fn supports_rewrite(&self) -> bool {
        true
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        let LogicalPlan::Join(join) = &plan else {
            return Ok(Transformed::no(plan));
        };

        // Check preconditions: inner join, no residual filter, has equijoin keys
        if !matches!(join.join_type, JoinType::Inner) {
            return Ok(Transformed::no(plan));
        }

        if join.filter.is_some() {
            return Ok(Transformed::no(plan));
        }

        if join.on.is_empty() {
            return Ok(Transformed::no(plan));
        }

        // Right side must be an aggregate
        let LogicalPlan::Aggregate(agg) = join.right.as_ref() else {
            return Ok(Transformed::no(plan));
        };

        // Get left input and aggregate input
        let left_input = join.left.as_ref();
        let agg_input = agg.input.as_ref();

        // Check if left and aggregate inputs are structurally equivalent
        if !plans_are_equivalent(left_input, agg_input) {
            return Ok(Transformed::no(plan));
        }

        // Try to rewrite
        match try_rewrite_to_window_and_projection(join, agg, left_input) {
            Ok(Some(new_plan)) => Ok(Transformed::yes(new_plan)),
            Ok(None) => Ok(Transformed::no(plan)),
            Err(_) => Ok(Transformed::no(plan)),
        }
    }
}

/// Check if two logical plans are structurally equivalent using display comparison.
fn plans_are_equivalent(p1: &LogicalPlan, p2: &LogicalPlan) -> bool {
    format!("{}", p1.display_indent()) == format!("{}", p2.display_indent())
}

/// Attempt to rewrite a join against an aggregate into a window aggregate with projection.
fn try_rewrite_to_window_and_projection(
    join: &Join,
    agg: &Aggregate,
    left_input: &LogicalPlan,
) -> Result<Option<LogicalPlan>> {
    // Extract join keys
    let join_keys: Vec<Expr> = join.on.iter().map(|(k, _)| k.clone()).collect();
    let join_right_keys: Vec<Expr> = join.on.iter().map(|(_, k)| k.clone()).collect();

    // Get aggregate group keys and expressions
    let agg_group_keys = &agg.group_expr;
    let agg_exprs = &agg.aggr_expr;

    // Check if all join keys are non-nullable
    let left_schema = left_input.schema();
    for join_key in &join_keys {
        if !is_expr_non_nullable(join_key, left_schema)? {
            return Ok(None);
        }
    }

    // Verify join keys match group keys exactly
    if !keys_match_exactly(&join_right_keys, agg_group_keys) {
        return Ok(None);
    }

    // Try to build window expressions from aggregate expressions
    let window_exprs = match build_window_expressions(agg_exprs, &join_keys)? {
        Some(exprs) => exprs,
        None => return Ok(None),
    };

    // Build output schema and projection
    let output_plan =
        build_window_output_projection(join, left_input, agg_group_keys, &window_exprs, agg)?;

    Ok(Some(output_plan))
}

/// Check if an expression is known to be non-nullable given a schema.
fn is_expr_non_nullable(expr: &Expr, schema: &DFSchema) -> Result<bool> {
    match expr {
        Expr::Column(col_ref) => {
            let result = schema
                .field_from_qualified_name(&col_ref.name)
                .ok()
                .map(|f| !f.is_nullable());
            Ok(result.unwrap_or(false))
        }
        Expr::Literal(_) => Ok(true),
        _ => Ok(false),
    }
}

/// Check if join keys match group keys exactly.
fn keys_match_exactly(join_keys: &[Expr], group_keys: &[Expr]) -> bool {
    if join_keys.len() != group_keys.len() {
        return false;
    }

    for (j_key, g_key) in join_keys.iter().zip(group_keys.iter()) {
        if !exprs_are_equal(j_key, g_key) {
            return false;
        }
    }

    true
}

/// Check if two expressions are semantically equal using display comparison.
fn exprs_are_equal(e1: &Expr, e2: &Expr) -> bool {
    format!("{}", e1.display()) == format!("{}", e2.display())
}

/// Build window expressions from aggregate expressions.
///
/// Returns None if any aggregate cannot be rewritten as a window aggregate.
fn build_window_expressions(
    agg_exprs: &[Expr],
    partition_keys: &[Expr],
) -> Result<Option<Vec<Expr>>> {
    let mut window_exprs = Vec::new();

    for agg_expr in agg_exprs {
        match try_to_window_expr(agg_expr, partition_keys)? {
            Some(window_expr) => window_exprs.push(window_expr),
            None => return Ok(None),
        }
    }

    Ok(Some(window_exprs))
}

/// Try to convert an aggregate expression to a window expression.
fn try_to_window_expr(agg_expr: &Expr, partition_keys: &[Expr]) -> Result<Option<Expr>> {
    match agg_expr {
        // Simple aggregate functions without DISTINCT or FILTER
        Expr::AggregateFunction(agg_func) if !agg_func.distinct && agg_func.filter.is_none() => {
            // Only support aggregates that work as window functions
            match &agg_func.func {
                AggregateFunction::Sum
                | AggregateFunction::Avg
                | AggregateFunction::Count
                | AggregateFunction::Min
                | AggregateFunction::Max => {
                    let window_expr = Expr::WindowFunction(
                        datafusion_expr::expr::WindowFunction {
                            fun: datafusion_expr::WindowFunctionDefinition::AggregateFunction(
                                agg_func.func.clone(),
                            ),
                            args: agg_func.args.clone(),
                            partition_by: partition_keys.to_vec(),
                            order_by: vec![],
                            window_frame: WindowFrame::new(None),
                            null_treatment: agg_func.null_treatment.clone(),
                        },
                    );
                    Ok(Some(window_expr))
                }
                _ => Ok(None),
            }
        }
        // Aliased aggregate
        Expr::Alias(alias) => {
            if let Some(window_expr) = try_to_window_expr(&alias.expr, partition_keys)? {
                Ok(Some(Expr::Alias(datafusion_expr::expr::Alias {
                    expr: Box::new(window_expr),
                    relation_name: alias.relation_name.clone(),
                    name: alias.name.clone(),
                })))
            } else {
                Ok(None)
            }
        }
        _ => Ok(None),
    }
}

/// Build the output projection to preserve schema and column ordering.
fn build_window_output_projection(
    join: &Join,
    left_input: &LogicalPlan,
    _agg_group_keys: &[Expr],
    window_exprs: &[Expr],
    _agg: &Aggregate,
) -> Result<LogicalPlan> {
    let join_schema = join.schema();
    let left_schema = left_input.schema();

    let mut projection_exprs = Vec::new();

    // Add all columns from the left input
    for field in left_schema.fields() {
        let col_name = field.qualified_name();
        projection_exprs.push(col(col_name));
    }

    // Add window aggregate expressions
    for window_expr in window_exprs {
        projection_exprs.push(window_expr.clone());
    }

    // Create a projection node with window expressions over the left input
    let projection = Projection {
        expr: projection_exprs,
        input: Arc::new(left_input.clone()),
        schema: join_schema.clone(),
        alias: None,
    };

    Ok(LogicalPlan::Projection(projection))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exprs_are_equal() {
        let expr1 = col("a");
        let expr2 = col("a");
        assert!(exprs_are_equal(&expr1, &expr2));
    }

    #[test]
    fn test_exprs_not_equal() {
        let expr1 = col("a");
        let expr2 = col("b");
        assert!(!exprs_are_equal(&expr1, &expr2));
    }

    #[test]
    fn test_keys_match_exactly() {
        let keys1 = vec![col("a"), col("b")];
        let keys2 = vec![col("a"), col("b")];
        assert!(keys_match_exactly(&keys1, &keys2));
    }

    #[test]
    fn test_keys_not_match_exactly_different_length() {
        let keys1 = vec![col("a")];
        let keys2 = vec![col("a"), col("b")];
        assert!(!keys_match_exactly(&keys1, &keys2));
    }

    #[test]
    fn test_keys_not_match_exactly_different_values() {
        let keys1 = vec![col("a"), col("b")];
        let keys2 = vec![col("a"), col("c")];
        assert!(!keys_match_exactly(&keys1, &keys2));
    }

    #[test]
    fn test_rule_name() {
        let rule = GroupByJoinToWindow::new();
        assert_eq!(rule.name(), "group_by_join_to_window");
    }

    #[test]
    fn test_apply_order() {
        let rule = GroupByJoinToWindow::new();
        assert_eq!(rule.apply_order(), Some(ApplyOrder::BottomUp));
    }

    #[test]
    fn test_supports_rewrite() {
        let rule = GroupByJoinToWindow::new();
        assert!(rule.supports_rewrite());
    }
}
