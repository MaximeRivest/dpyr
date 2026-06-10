from __future__ import annotations

import duckdb
import polars as pl
import pytest

import dpyr as d

_con = duckdb.connect()
_table_counter = [0]


def make_polars(data: dict) -> d.DFrame:
    return d.from_polars(pl.DataFrame(data))


def make_duckdb(data: dict) -> d.DFrame:
    _table_counter[0] += 1
    name = f"t{_table_counter[0]}"
    _con.register(f"{name}_arrow", pl.DataFrame(data).to_arrow())
    _con.execute(f'CREATE TABLE "{name}" AS SELECT * FROM "{name}_arrow"')
    _con.unregister(f"{name}_arrow")
    return d.from_duckdb(_con, name)


def both(data: dict) -> list[d.DFrame]:
    return [make_polars(data), make_duckdb(data)]


@pytest.fixture(params=["polars", "duckdb"])
def backend(request):
    return request.param


@pytest.fixture
def make(backend):
    return make_polars if backend == "polars" else make_duckdb


STARWARS = {
    "name": ["Luke", "Leia", "Han", "Chewie", "R2D2", "Yoda"],
    "height": [172, 150, 180, 228, 96, 66],
    "mass": [77.0, 49.0, 80.0, 112.0, None, 17.0],
    "species": ["Human", "Human", "Human", "Wookiee", "Droid", None],
}
