# -*- coding: utf-8 -*-
"""Register E59 (28/10/2026): at a fixed ubatch, the logical batch (-b) changes answers too, so it is part of the
exactness reference. Every profile records the -b it was measured at; a launch whose -b would differ is refused, and so
are LLAMA_ARG_BATCH / LLAMA_ARG_UBATCH in the environment (llama-server would read them without a flag)."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from marian import launch, machine, profile
from marian.engines import parse_help
from marian.profile import sha256_head

import test_m13 as m13
import test_prefill_ubatch as pu

BATCH_LINE = ("-b,    --batch-size N                   logical maximum batch size (default: {})\n"
              "                                        (env: LLAMA_ARG_BATCH)\n")


def help_with_batch(n):
    return pu.HELP_0009 + BATCH_LINE.format(n)


class BatchFake(pu.PrefillFake):
    def __init__(self, default_batch=2048):
        super().__init__(help_text=help_with_batch(default_batch))


def check(doc, help_text, ubatch=512, row=0):
    with tempfile.TemporaryDirectory() as d, m13.fake_machine(None):
        model = m13.toy_moe(Path(d))
        doc = json.loads(json.dumps(doc))
        doc["machine"]["fingerprint"], doc["model"]["id"] = machine.detect().fingerprint(), sha256_head(model)
        m13.FakeLlama.help_text, m13.FakeLlama.version_text = help_text, pu.VERSION
        return launch.check(doc["rows"][row], doc, model, sha256_head(model), m13.FakeLlama("x"),
                            vram_free_mib=9000, ubatch=ubatch)


class Batch(unittest.TestCase):
    def test_help_states_the_default_batch(self):
        self.assertEqual(parse_help(help_with_batch(2048))["default_batch"], 2048)
        self.assertIsNone(parse_help(pu.HELP_0009)["default_batch"])

    def test_profile_records_the_batch_it_ran_at(self):
        doc, _ = pu.run_profile(BatchFake(2048))
        self.assertEqual(doc["batch"], 2048)                         # the default ubatch: llama-server's own -b
        doc, _ = pu.run_profile(BatchFake(2048), ubatch=1024)
        self.assertEqual(doc["batch"], 1024)                         # a declared ubatch N runs -ub N -b N

    def test_a_changed_default_batch_is_refused(self):
        doc = dict(pu.DOC_512, batch=2048)
        reasons, _ = check(doc, help_with_batch(2048))
        self.assertFalse([r for r in reasons if r.startswith("batch:")], reasons)
        reasons, _ = check(doc, help_with_batch(4096))
        self.assertTrue(any(r.startswith("batch: the profile was measured at -b 2048") for r in reasons), reasons)

    def test_a_declared_ubatch_checks_its_own_batch(self):
        doc = dict(pu.DOC_2048, batch=2048)
        reasons, _ = check(doc, help_with_batch(4096), ubatch=2048)  # -b follows the declared ubatch, not the default
        self.assertFalse([r for r in reasons if r.startswith("batch:")], reasons)

    def test_an_unreadable_default_is_refused_and_an_old_profile_warned(self):
        reasons, _ = check(dict(pu.DOC_512, batch=2048), pu.HELP_0009)
        self.assertTrue(any("does not state its default -b" in r for r in reasons), reasons)
        reasons, warns = check(pu.DOC_512, help_with_batch(2048))
        self.assertFalse([r for r in reasons if r.startswith("batch:")], reasons)
        self.assertTrue(any("does not record its logical batch" in w for w in warns), warns)

    def test_batch_variables_in_the_environment_are_refused(self):
        for var in profile.BATCH_ENV:
            with self.subTest(var=var), mock.patch.dict(os.environ, {var: "512"}):
                reasons, _ = check(dict(pu.DOC_512, batch=2048), help_with_batch(2048))
                self.assertTrue(any(r.startswith("environment: " + var) for r in reasons), reasons)
                eng = BatchFake()
                with self.assertRaises(ValueError):
                    pu.run_profile(eng)
                self.assertEqual(eng.started, [])


if __name__ == "__main__":
    unittest.main()
