"""DFrame / GroupedDFrame: the user-facing verb surface (ROADMAP 1.4).

A DFrame is an immutable handle on a logical plan. Verbs return new frames;
schema validation happens inside the plan node constructors, so mistakes
raise here, on the user's line. Execution arrives with the backends
(Epics 2/5); the materialization model (Epic 3) hangs collection off this
class.
"""

from __future__ import annotations

from typing import Literal

from . import plan as p
from .dtypes import DType
from .errors import GroupError
from .expr import Col, Desc, Expr

ColRef = str | Col


def _name(ref: ColRef, what: str) -> str:
    if isinstance(ref, Col):
        return ref.name
    if isinstance(ref, str):
        return ref
    raise TypeError(f"{what} expects column names or col.<name>, got {type(ref).__name__}")


class DFrame:
    """An ungrouped tidy frame (lazy plan + eager schema)."""

    def __init__(self, plan_node: p.PlanNode) -> None:
        if plan_node.groups:
            raise GroupError("internal: DFrame built from a grouped plan")
        self._plan = plan_node

    # -- metadata ------------------------------------------------------
    @property
    def plan(self) -> p.PlanNode:
        return self._plan

    @property
    def schema(self) -> dict[str, DType]:
        return dict(self._plan.schema)

    @property
    def columns(self) -> list[str]:
        return list(self._plan.schema)

    def __dir__(self) -> list[str]:  # runtime completion (ROADMAP 7.1)
        return [*super().__dir__(), *self._plan.schema]

    def __repr__(self) -> str:
        cols = ", ".join(f"{k} <{v!r}>" for k, v in self._plan.schema.items())
        return f"<DFrame [{cols}] (lazy; backends arrive in Epic 2)>"

    def _wrap(self, node: p.PlanNode) -> DFrame | GroupedDFrame:
        return GroupedDFrame(node) if node.groups else DFrame(node)

    # -- verbs ----------------------------------------------------------
    def filter(self, *predicates: Expr) -> DFrame:
        return DFrame(p.Filter(self._plan, predicates))

    def mutate(self, **exprs: Expr) -> DFrame:
        return DFrame(p.Mutate(self._plan, tuple(exprs.items())))

    def select(self, *cols: ColRef) -> DFrame:
        return DFrame(p.Select(self._plan, tuple(_name(c, "select()") for c in cols)))

    def rename(self, **mapping: ColRef) -> DFrame:
        pairs = tuple((new, _name(old, "rename()")) for new, old in mapping.items())
        return DFrame(p.Rename(self._plan, pairs))

    def arrange(self, *keys: Expr | Desc) -> DFrame:
        return DFrame(p.Arrange(self._plan, keys))

    def distinct(self, *cols: ColRef) -> DFrame:
        return DFrame(p.Distinct(self._plan, tuple(_name(c, "distinct()") for c in cols)))

    def slice_head(self, n: int = 5) -> DFrame:
        return DFrame(p.Slice(self._plan, "head", n))

    def slice_tail(self, n: int = 5) -> DFrame:
        return DFrame(p.Slice(self._plan, "tail", n))

    def slice_sample(self, n: int = 5) -> DFrame:
        return DFrame(p.Slice(self._plan, "sample", n))

    def group_by(self, *keys: ColRef) -> GroupedDFrame:
        return GroupedDFrame(p.GroupBy(self._plan, tuple(_name(k, "group_by()") for k in keys)))

    def summarize(self, **aggs: Expr) -> DFrame:
        return DFrame(p.Summarize(self._plan, tuple(aggs.items())))

    summarise = summarize

    def count(self, *cols: ColRef, name: str = "n") -> DFrame:
        from .expr import n as n_
        if not cols:
            return self.summarize(**{name: n_()})
        grouped = self.group_by(*cols)
        return grouped.summarize(**{name: n_()})

    # -- joins -----------------------------------------------------------
    def _join(self, other: DFrame, how: p.JoinHow, on: ColRef | list[ColRef],
              suffix: tuple[str, str], na_matches: Literal["na", "never"]) -> DFrame:
        if isinstance(other, GroupedDFrame):
            raise GroupError("joining a grouped frame is not supported; ungroup() it first")
        refs = on if isinstance(on, list) else [on]
        keys = tuple(_name(r, "join on=") for r in refs)
        return DFrame(p.Join(self._plan, other._plan, how, keys, suffix, na_matches))

    def inner_join(self, other: DFrame, on: ColRef | list[ColRef],
                   suffix: tuple[str, str] = (".x", ".y"),
                   na_matches: Literal["na", "never"] = "na") -> DFrame:
        return self._join(other, "inner", on, suffix, na_matches)

    def left_join(self, other: DFrame, on: ColRef | list[ColRef],
                  suffix: tuple[str, str] = (".x", ".y"),
                  na_matches: Literal["na", "never"] = "na") -> DFrame:
        return self._join(other, "left", on, suffix, na_matches)

    def right_join(self, other: DFrame, on: ColRef | list[ColRef],
                   suffix: tuple[str, str] = (".x", ".y"),
                   na_matches: Literal["na", "never"] = "na") -> DFrame:
        return self._join(other, "right", on, suffix, na_matches)

    def full_join(self, other: DFrame, on: ColRef | list[ColRef],
                  suffix: tuple[str, str] = (".x", ".y"),
                  na_matches: Literal["na", "never"] = "na") -> DFrame:
        return self._join(other, "full", on, suffix, na_matches)

    def semi_join(self, other: DFrame, on: ColRef | list[ColRef]) -> DFrame:
        return self._join(other, "semi", on, (".x", ".y"), "na")

    def anti_join(self, other: DFrame, on: ColRef | list[ColRef]) -> DFrame:
        return self._join(other, "anti", on, (".x", ".y"), "na")

    # -- reshaping ---------------------------------------------------------
    def pivot_longer(self, cols: list[ColRef], names_to: str = "name",
                     values_to: str = "value") -> DFrame:
        keep = tuple(_name(c, "pivot_longer()") for c in cols)
        return DFrame(p.PivotLonger(self._plan, keep, names_to, values_to))

    def pivot_wider(self, names_from: ColRef, values_from: ColRef) -> DFrame:
        return DFrame(p.PivotWider(self._plan, _name(names_from, "pivot_wider()"),
                                   _name(values_from, "pivot_wider()")))


class GroupedDFrame(DFrame):
    """A grouped frame: its own type so completion and semantics differ."""

    def __init__(self, plan_node: p.PlanNode) -> None:
        if not plan_node.groups:
            raise GroupError("internal: GroupedDFrame built from an ungrouped plan")
        self._plan = plan_node

    @property
    def groups(self) -> tuple[str, ...]:
        return self._plan.groups

    def __repr__(self) -> str:
        cols = ", ".join(f"{k} <{v!r}>" for k, v in self._plan.schema.items())
        return (f"<GroupedDFrame groups=({', '.join(self._plan.groups)}) "
                f"[{cols}] (lazy)>")

    def ungroup(self) -> DFrame:
        # plan-level no-op: grouping is metadata
        return DFrame(_regroup(self._plan, ()))

    def filter(self, *predicates: Expr) -> GroupedDFrame:
        return GroupedDFrame(p.Filter(self._plan, predicates))

    def mutate(self, **exprs: Expr) -> GroupedDFrame:
        return GroupedDFrame(p.Mutate(self._plan, tuple(exprs.items())))

    def group_by(self, *keys: ColRef) -> GroupedDFrame:
        names = tuple(_name(k, "group_by()") for k in keys)
        return GroupedDFrame(p.GroupBy(self._plan, self._plan.groups + names))

    def summarize(self, **aggs: Expr) -> DFrame | GroupedDFrame:
        node = p.Summarize(self._plan, tuple(aggs.items()))
        return GroupedDFrame(node) if node.groups else DFrame(node)

    summarise = summarize

    def arrange(self, *keys: Expr | Desc) -> GroupedDFrame:
        return GroupedDFrame(p.Arrange(self._plan, keys))

    def select(self, *cols: ColRef) -> GroupedDFrame:
        return GroupedDFrame(p.Select(self._plan, tuple(_name(c, "select()") for c in cols)))


def _regroup(node: p.PlanNode, groups: tuple[str, ...]) -> p.PlanNode:
    """Re-tag a plan with different active groups without changing data ops."""
    import copy
    clone = copy.copy(node)
    object.__setattr__(clone, "groups", groups)
    return clone


def from_schema(schema: dict[str, DType], name: str = "table") -> DFrame:
    """Build a frame from a known schema — the Epic-1 source; file/db
    sources arrive with the backends."""
    return DFrame(p.Source(name, tuple(schema.items())))
