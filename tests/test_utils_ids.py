import unittest

from src.utils import is_whitelisted_id, normalize_id, normalize_id_set


class IdNormalizationTests(unittest.TestCase):
    def test_normalize_id(self):
        self.assertEqual(normalize_id(123), "123")
        self.assertEqual(normalize_id("  !room:server  "), "!room:server")
        self.assertIsNone(normalize_id(True))
        self.assertIsNone(normalize_id(""))

    def test_normalize_id_set_mixed_values(self):
        values = [123, " 456 ", "!room:server", "", False, None]
        self.assertEqual(normalize_id_set(values), {"123", "456", "!room:server"})

    def test_is_whitelisted_id_with_numeric_and_string(self):
        whitelist = [123, "#ai", "!room:server"]
        self.assertTrue(is_whitelisted_id(123, whitelist))
        self.assertTrue(is_whitelisted_id("#ai", whitelist))
        self.assertTrue(is_whitelisted_id("!room:server", whitelist))
        self.assertFalse(is_whitelisted_id("999", whitelist))

