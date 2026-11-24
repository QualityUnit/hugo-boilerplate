from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).parents[1]))
import taxonomy_keys  # noqa: E402


SOURCE = '''+++
title = "Best platforms for affiliate marketing videos"
url = "/faq/best-platforms-affiliate-marketing-videos/"
author = "Post Affiliate Pro Team"
faq-categories = ["Traffic and Promotion Channels"]
blog-categories = [
  "Affiliate Marketing",
  "Tutorials and Guides",
]

[[faq]]
question = "Which platform?"
answer = "YouTube."
+++
Body
'''

TRANSLATED = '''+++
title = "Beste Plattformen für Affiliate-Marketing-Videos"
url = "/faq/best-platforms-affiliate-marketing-videos/"
author = "Post Affiliate Pro Team"
faq-categories = ["Traffic- und Promotion-Kanäle"]
blog-categories = [
  "Affiliate-Marketing",
  "Tutorials und Anleitungen",
]

[[faq]]
question = "Welche Plattform?"
answer = "YouTube."
+++
Inhalt
'''

KEYS = ['author', 'blog-categories', 'faq-categories']


class RestoreTaxonomiesTests(unittest.TestCase):
    def test_translated_values_get_the_english_ones(self):
        text, restored = taxonomy_keys.restore_taxonomies(TRANSLATED, SOURCE, KEYS)
        self.assertEqual(['blog-categories', 'faq-categories'], restored)
        self.assertIn('faq-categories = ["Traffic and Promotion Channels"]', text)
        self.assertIn('  "Affiliate Marketing",\n  "Tutorials and Guides",\n]', text)
        self.assertNotIn('Kanäle', text)
        self.assertNotIn('Anleitungen', text)

    def test_everything_else_is_kept(self):
        text, _ = taxonomy_keys.restore_taxonomies(TRANSLATED, SOURCE, KEYS)
        self.assertIn('title = "Beste Plattformen für Affiliate-Marketing-Videos"', text)
        self.assertIn('question = "Welche Plattform?"', text)
        self.assertTrue(text.endswith('+++\nInhalt\n'))

    def test_matching_values_leave_the_text_untouched(self):
        text, restored = taxonomy_keys.restore_taxonomies(SOURCE, SOURCE, KEYS)
        self.assertEqual(SOURCE, text)
        self.assertEqual([], restored)

    def test_missing_key_is_added_before_the_first_table(self):
        translated = TRANSLATED.replace('faq-categories = ["Traffic- und Promotion-Kanäle"]\n', '')
        text, restored = taxonomy_keys.restore_taxonomies(translated, SOURCE, ['faq-categories'])
        self.assertEqual(['faq-categories'], restored)
        self.assertLess(text.index('faq-categories ='), text.index('[[faq]]'))
        frontmatter = text.split('+++')[1]
        data = taxonomy_keys._toml_loads(frontmatter)
        self.assertEqual(['Traffic and Promotion Channels'], data['faq-categories'])
        self.assertNotIn('faq-categories', data['faq'][0])

    def test_key_the_source_lacks_is_left_alone(self):
        source = SOURCE.replace('author = "Post Affiliate Pro Team"\n', '')
        translated = TRANSLATED.replace('author = "Post Affiliate Pro Team"',
                                        'author = "Équipe Post Affiliate Pro"')
        text, _ = taxonomy_keys.restore_taxonomies(translated, source, ['author'])
        self.assertIn('author = "Équipe Post Affiliate Pro"', text)

    def test_broken_translated_frontmatter_is_returned_unchanged(self):
        broken = TRANSLATED.replace('title = "Beste', 'title = Beste')
        text, restored = taxonomy_keys.restore_taxonomies(broken, SOURCE, KEYS)
        self.assertEqual(broken, text)
        self.assertEqual([], restored)

    def test_bom_and_crlf_are_kept(self):
        translated = '﻿' + TRANSLATED.replace('\n', '\r\n')
        text, restored = taxonomy_keys.restore_taxonomies(translated, SOURCE, ['faq-categories'])
        self.assertEqual(['faq-categories'], restored)
        self.assertTrue(text.startswith('﻿+++'))
        self.assertIn('faq-categories = ["Traffic and Promotion Channels"]', text)


class LoadTaxonomyKeysTests(unittest.TestCase):
    def setUp(self):
        taxonomy_keys._KEYS_CACHE.clear()

    def _site(self, files):
        root = Path(tempfile.mkdtemp())
        for rel_path, content in files.items():
            path = root / rel_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding='utf-8')
        return root

    def test_reads_taxonomies_toml(self):
        root = self._site({'config/_default/taxonomies.toml':
                           'author = "author"\nfaq-category = "faq-categories"\n'})
        self.assertEqual(['author', 'faq-categories'], taxonomy_keys.load_taxonomy_keys(root))

    def test_reads_the_taxonomies_table_of_hugo_toml(self):
        root = self._site({'hugo.toml': 'title = "x"\n[taxonomies]\ntag = "tags"\n'})
        self.assertEqual(['tags'], taxonomy_keys.load_taxonomy_keys(root))

    def test_empty_config_falls_back_to_hugo_defaults(self):
        root = self._site({'config/_default/taxonomies.toml': '# none\n'})
        self.assertEqual(['tags', 'categories'], taxonomy_keys.load_taxonomy_keys(root))


if __name__ == '__main__':
    unittest.main()
