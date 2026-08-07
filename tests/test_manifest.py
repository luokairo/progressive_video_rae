from pathlib import Path

from progressive_video_rae.data.manifest import (
    CsvSource,
    assign_split,
    merge_source_rows,
    parse_csv_spec,
)


def test_csv_spec_preserves_three_source_meanings(tmp_path: Path):
    paths = [tmp_path / f"{name}.csv" for name in ("human", "environment", "music")]
    spec = tmp_path / "data_csv.md"
    spec.write_text(
        f"{paths[0]}\n人物\n\n{paths[1]}\n环境音\n\n{paths[2]}\n音乐\n",
        encoding="utf-8",
    )
    sources = parse_csv_spec(spec)
    assert [(source.category, source.source_tag) for source in sources] == [
        ("human", "human"),
        ("non_speech", "environment"),
        ("non_speech", "music"),
    ]


def test_environment_music_overlap_merges_tags_and_human_wins(tmp_path: Path):
    human = CsvSource(tmp_path / "h.csv", "human", "human", "")
    environment = CsvSource(tmp_path / "e.csv", "non_speech", "environment", "")
    music = CsvSource(tmp_path / "m.csv", "non_speech", "music", "")
    rows = merge_source_rows(
        [
            (human, [{"path": "/v/h.mp4", "caption": "person"}]),
            (
                environment,
                [
                    {"path": "/v/shared.mp4", "caption": "short"},
                    {"path": "/v/h.mp4", "caption": "wrong category"},
                ],
            ),
            (music, [{"path": "/v/shared.mp4", "caption": "a longer caption"}]),
        ]
    )
    by_path = {row["path"]: row for row in rows}
    assert by_path["/v/shared.mp4"]["source_tags"] == ["environment", "music"]
    assert by_path["/v/shared.mp4"]["caption"] == "a longer caption"
    assert by_path["/v/h.mp4"]["category"] == "human"


def test_hash_split_is_deterministic():
    first = assign_split("/video/a.mp4", 20260807, (0.95, 0.025, 0.025))
    second = assign_split("/video/a.mp4", 20260807, (0.95, 0.025, 0.025))
    assert first == second

