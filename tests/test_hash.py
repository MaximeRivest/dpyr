from dpyr import col, from_schema, n, plan_hash
from dpyr import INT64, STR

sw = from_schema({"name": STR, "height": INT64}, name="starwars")


def test_same_plan_same_hash():
    a = sw.filter(col.height > 180).group_by(col.name).summarize(n=n())
    b = sw.filter(col.height > 180).group_by(col.name).summarize(n=n())
    assert plan_hash(a.plan) == plan_hash(b.plan)


def test_different_predicate_different_hash():
    a = sw.filter(col.height > 180)
    b = sw.filter(col.height > 181)
    assert plan_hash(a.plan) != plan_hash(b.plan)


def test_verb_order_matters():
    a = sw.filter(col.height > 180).select(col.name)
    b = sw.select(col.name, col.height).filter(col.height > 180)
    assert plan_hash(a.plan) != plan_hash(b.plan)


def test_source_name_matters():
    other = from_schema({"name": STR, "height": INT64}, name="other")
    assert plan_hash(sw.plan) != plan_hash(other.plan)
