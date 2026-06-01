"""
Unit tests for src/data/ffpp_splits.py

Tests cover:
- parse_filename: all known FF++ naming conventions
- parse_filename: raises on unknown filenames
- create_splits: determinism (same result on every call)
- create_splits: pool sizes match expected counts
- assign_split: returns 'train' / 'val' / 'test' for known actors
- assign_split: consistency — every actor in the dataset gets exactly one split
- get_method_from_path: extracts method name from FF++ directory structure
"""

from __future__ import annotations

import pytest

from src.data.ffpp_splits import (
    assign_split,
    create_splits,
    get_method_from_path,
    parse_filename,
)


# ──────────────────────────────────────────────────────────────────────────────
# parse_filename
# ──────────────────────────────────────────────────────────────────────────────

class TestParseFilename:
    def test_youtube_real_3digit(self):
        r = parse_filename("001.mp4")
        assert r["type"] == "real"
        assert r["actor_1"] == "001"
        assert r["source"] == "youtube"

    def test_actors_real_2digit(self):
        r = parse_filename("01.mp4")
        assert r["type"] == "real"
        assert r["actor_1"] == "01"
        assert r["source"] == "actors"

    def test_youtube_deepfake(self):
        r = parse_filename("001_002.mp4")
        assert r["type"] == "deepfake"
        assert r["actor_1"] == "001"
        assert r["actor_2"] == "002"
        assert r["source"] == "youtube"

    def test_actors_deepfake_double_underscore(self):
        # Actual FF++ format: "01_02__scene1.mp4"
        # parts[0] after split("__") is "01_02", actors = ["01", "02"]
        r = parse_filename("01_02__scene1.mp4")
        assert r["type"] == "deepfake"
        assert r["actor_1"] == "01"
        assert r["actor_2"] == "02"
        assert r["source"] == "actors"
        assert r["scene"] == "scene1"

    def test_actors_real_double_underscore(self):
        r = parse_filename("03__scene1__uid123.mp4")
        assert r["type"] == "real"
        assert r["actor_1"] == "03"
        assert r["source"] == "actors"

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            parse_filename("unknown_video_name.mp4")

    def test_cache_consistent(self):
        r1 = parse_filename("001.mp4")
        r2 = parse_filename("001.mp4")
        assert r1 == r2


# ──────────────────────────────────────────────────────────────────────────────
# create_splits
# ──────────────────────────────────────────────────────────────────────────────

class TestCreateSplits:
    def test_deterministic(self):
        s1 = create_splits()
        s2 = create_splits()
        assert s1["train_actors"] == s2["train_actors"]
        assert s1["test_youtube"] == s2["test_youtube"]

    def test_actor_pool_sizes(self):
        s = create_splits()
        assert len(s["train_actors"]) == 20
        assert len(s["val_actors"])   == 4
        assert len(s["test_actors"])  == 4

    def test_youtube_pool_sizes(self):
        s = create_splits()
        assert len(s["train_youtube"]) == 700
        assert len(s["val_youtube"])   == 150
        assert len(s["test_youtube"])  == 150

    def test_actor_pools_disjoint(self):
        s = create_splits()
        pools = [s["train_actors"], s["val_actors"], s["test_actors"]]
        all_ids = set()
        for p in pools:
            assert all_ids.isdisjoint(p), "actor split pools overlap"
            all_ids |= p

    def test_youtube_pools_disjoint(self):
        s = create_splits()
        pools = [s["train_youtube"], s["val_youtube"], s["test_youtube"]]
        all_ids = set()
        for p in pools:
            assert all_ids.isdisjoint(p), "youtube split pools overlap"
            all_ids |= p


# ──────────────────────────────────────────────────────────────────────────────
# assign_split
# ──────────────────────────────────────────────────────────────────────────────

class TestAssignSplit:
    def setup_method(self):
        self.splits = create_splits()

    def test_youtube_real_gets_a_split(self):
        # Use a known youtube ID from the train pool
        train_id = next(iter(self.splits["train_youtube"]))
        vpath = f"/data/original_sequences/youtube/{train_id}.mp4"
        assert assign_split(vpath, self.splits) == "train"

    def test_youtube_test_id(self):
        test_id = next(iter(self.splits["test_youtube"]))
        vpath = f"/data/original_sequences/youtube/{test_id}.mp4"
        assert assign_split(vpath, self.splits) == "test"

    def test_actors_real_gets_a_split(self):
        train_actor = next(iter(self.splits["train_actors"]))
        vpath = f"/data/original_sequences/actors/{train_actor}.mp4"
        result = assign_split(vpath, self.splits)
        assert result == "train"

    def test_actors_val(self):
        val_actor = next(iter(self.splits["val_actors"]))
        vpath = f"/data/original_sequences/actors/{val_actor}.mp4"
        assert assign_split(vpath, self.splits) == "val"

    def test_deepfake_split_follows_target_actor(self):
        # actor_2 is the target; split should match actor_2's pool.
        # FF++ actors deepfake format: "{actor_1}_{actor_2}__scene.mp4"
        # (single underscore between actors, double before scene)
        test_actor  = next(iter(self.splits["test_actors"]))
        train_actor = next(iter(self.splits["train_actors"]))
        vpath = f"/data/manipulated_sequences/Deepfakes/{train_actor}_{test_actor}__scene1.mp4"
        assert assign_split(vpath, self.splits) == "test"

    def test_unknown_filename_returns_none(self):
        vpath = "/data/some_dir/??????.mp4"
        result = assign_split(vpath, self.splits)
        assert result is None

    def test_all_actors_covered(self):
        """Every actor ID 01–28 gets exactly one split."""
        all_actors = {f"{i:02d}" for i in range(1, 29)}
        s = self.splits
        covered = s["train_actors"] | s["val_actors"] | s["test_actors"]
        assert covered == all_actors


# ──────────────────────────────────────────────────────────────────────────────
# get_method_from_path
# ──────────────────────────────────────────────────────────────────────────────

class TestGetMethodFromPath:
    def test_deepfakes_method(self):
        p = "/data/manipulated_sequences/Deepfakes/01_02.mp4"
        assert get_method_from_path(p) == "Deepfakes"

    def test_neuraltextures(self):
        p = "/data/manipulated_sequences/NeuralTextures/001_002.mp4"
        assert get_method_from_path(p) == "NeuralTextures"

    def test_original(self):
        p = "/data/original_sequences/youtube/001.mp4"
        assert get_method_from_path(p) == "original"

    def test_windows_style_separators(self):
        p = r"C:\data\manipulated_sequences\FaceSwap\01_02.mp4"
        assert get_method_from_path(p) == "FaceSwap"
