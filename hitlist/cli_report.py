# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

"""``hitlist report``: data quality report."""

from __future__ import annotations

import argparse


def _report(args: argparse.Namespace) -> None:
    from .report import run_report

    text = run_report(
        mhc_class=args.mhc_class,
        output=args.output,
        from_csv=getattr(args, "from_csv", False),
    )
    if not args.output:
        print(text)


def register(sub: argparse._SubParsersAction) -> None:
    """Register ``hitlist report``."""
    p_report = sub.add_parser(
        "report",
        help=(
            "Data quality report.  Reads observations.parquet by default "
            "(instant); pass --from-csv to fall back to a live raw-CSV scan."
        ),
    )
    p_report.add_argument(
        "--class", dest="mhc_class", help="MHC class filter (I, II, or non-classical)"
    )
    p_report.add_argument(
        "--from-csv",
        action="store_true",
        help=(
            "Scan the raw IEDB/CEDAR CSVs instead of reading observations.parquet. "
            "Slow (minutes); useful before `hitlist build observations` has run."
        ),
    )
    p_report.add_argument("--output", "-o", help="Save report to file")
    p_report.set_defaults(func=_report)
