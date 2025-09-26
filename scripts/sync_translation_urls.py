#!/usr/bin/env python3
"""
Theme-level script to sync translation URLs with directory structure.
Ensures that files in subdirectories have proper URL paths that match their directory structure.

For example, if content/sk/about/_index.md has url = "/o-nas/", 
then all files in that directory should have URLs like "/o-nas/filename/"

This is a generic theme script that can be used across different Hugo projects.
"""

import os
import re
import toml
from pathlib import Path
from typing import Dict, Optional, Tuple
import sys
import argparse

def get_content_dir(hugo_root: Optional[str] = None) -> Path:
    """Get the content directory path"""
    if hugo_root:
        return Path(hugo_root) / 'content'
    
    # Try to find content dir relative to script location
    # This script is in themes/boilerplate/scripts/
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    
    # Check if we're in a theme directory
    if 'themes' in script_dir.parts:
        # Go up to the Hugo root
        theme_index = script_dir.parts.index('themes')
        hugo_root = Path(*script_dir.parts[:theme_index])
        return hugo_root / 'content'
    
    # Fallback to relative path
    return Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../content')))

# Language-specific directory name translations
# Maps language -> directory_name -> translated_name
DIRECTORY_TRANSLATIONS = {
    'sk': {
        'about': 'o-nas',
        'features': 'funkcie',
        'pricing': 'ceny',
        'support': 'podpora',
        'blog': 'blog',
        'resources': 'zdroje',
        'integrations': 'integracie',
        'templates': 'sablony',
        'affiliate-manager': 'affiliate-manazer',
        'glossary': 'slovnik',
        'academy': 'akademia',
        'comparisons': 'porovnania',
        'directory': 'adresar',
        'reviews': 'recenzie'
    },
    'cs': {
        'about': 'o-nas',
        'features': 'funkce',
        'pricing': 'cenik',
        'support': 'podpora',
        'blog': 'blog',
        'resources': 'zdroje',
        'integrations': 'integrace',
        'templates': 'sablony',
        'affiliate-manager': 'affiliate-manazer',
        'glossary': 'slovnik',
        'academy': 'akademie',
        'comparisons': 'srovnani',
        'directory': 'adresar',
        'reviews': 'recenze'
    },
    'de': {
        'about': 'ueber-uns',
        'features': 'funktionen',
        'pricing': 'preise',
        'support': 'unterstuetzung',
        'blog': 'blog',
        'resources': 'ressourcen',
        'integrations': 'integrationen',
        'templates': 'vorlagen',
        'affiliate-manager': 'affiliate-manager',
        'glossary': 'glossar',
        'academy': 'akademie',
        'comparisons': 'vergleiche',
        'directory': 'verzeichnis',
        'reviews': 'bewertungen'
    },
    'fr': {
        'about': 'a-propos',
        'features': 'fonctionnalites',
        'pricing': 'tarifs',
        'support': 'support',
        'blog': 'blog',
        'resources': 'ressources',
        'integrations': 'integrations',
        'templates': 'modeles',
        'affiliate-manager': 'gestionnaire-affilie',
        'glossary': 'glossaire',
        'academy': 'academie',
        'comparisons': 'comparaisons',
        'directory': 'repertoire',
        'reviews': 'avis'
    },
    'es': {
        'about': 'acerca-de',
        'features': 'caracteristicas',
        'pricing': 'precios',
        'support': 'soporte',
        'blog': 'blog',
        'resources': 'recursos',
        'integrations': 'integraciones',
        'templates': 'plantillas',
        'affiliate-manager': 'gerente-afiliado',
        'glossary': 'glosario',
        'academy': 'academia',
        'comparisons': 'comparaciones',
        'directory': 'directorio',
        'reviews': 'resenas'
    },
    'it': {
        'about': 'chi-siamo',
        'features': 'caratteristiche',
        'pricing': 'prezzi',
        'support': 'supporto',
        'blog': 'blog',
        'resources': 'risorse',
        'integrations': 'integrazioni',
        'templates': 'modelli',
        'affiliate-manager': 'gestore-affiliato',
        'glossary': 'glossario',
        'academy': 'accademia',
        'comparisons': 'confronti',
        'directory': 'directory',
        'reviews': 'recensioni'
    },
    'pt': {
        'about': 'sobre',
        'features': 'recursos',
        'pricing': 'precos',
        'support': 'suporte',
        'blog': 'blog',
        'resources': 'recursos',
        'integrations': 'integracoes',
        'templates': 'modelos',
        'affiliate-manager': 'gerente-afiliado',
        'glossary': 'glossario',
        'academy': 'academia',
        'comparisons': 'comparacoes',
        'directory': 'diretorio',
        'reviews': 'avaliacoes'
    },
    'pt-br': {
        'about': 'sobre',
        'features': 'recursos',
        'pricing': 'precos',
        'support': 'suporte',
        'blog': 'blog',
        'resources': 'recursos',
        'integrations': 'integracoes',
        'templates': 'modelos',
        'affiliate-manager': 'gerente-afiliado',
        'glossary': 'glossario',
        'academy': 'academia',
        'comparisons': 'comparacoes',
        'directory': 'diretorio',
        'reviews': 'avaliacoes'
    },
    'nl': {
        'about': 'over',
        'features': 'functies',
        'pricing': 'prijzen',
        'support': 'ondersteuning',
        'blog': 'blog',
        'resources': 'middelen',
        'integrations': 'integraties',
        'templates': 'sjablonen',
        'affiliate-manager': 'affiliate-manager',
        'glossary': 'woordenlijst',
        'academy': 'academie',
        'comparisons': 'vergelijkingen',
        'directory': 'directory',
        'reviews': 'beoordelingen'
    },
    'da': {
        'about': 'om',
        'features': 'funktioner',
        'pricing': 'priser',
        'support': 'support',
        'blog': 'blog',
        'resources': 'ressourcer',
        'integrations': 'integrationer',
        'templates': 'skabeloner',
        'affiliate-manager': 'affiliate-manager',
        'glossary': 'ordliste',
        'academy': 'akademi',
        'comparisons': 'sammenligninger',
        'directory': 'oversigt',
        'reviews': 'anmeldelser'
    },
    'sv': {
        'about': 'om',
        'features': 'funktioner',
        'pricing': 'priser',
        'support': 'support',
        'blog': 'blogg',
        'resources': 'resurser',
        'integrations': 'integrationer',
        'templates': 'mallar',
        'affiliate-manager': 'affiliate-hanterare',
        'glossary': 'ordlista',
        'academy': 'akademi',
        'comparisons': 'jamforelser',
        'directory': 'katalog',
        'reviews': 'recensioner'
    },
    'no': {
        'about': 'om',
        'features': 'funksjoner',
        'pricing': 'priser',
        'support': 'støtte',
        'blog': 'blogg',
        'resources': 'ressurser',
        'integrations': 'integrasjoner',
        'templates': 'maler',
        'affiliate-manager': 'affiliate-leder',
        'glossary': 'ordliste',
        'academy': 'akademi',
        'comparisons': 'sammenligninger',
        'directory': 'katalog',
        'reviews': 'anmeldelser'
    },
    'fi': {
        'about': 'tietoja',
        'features': 'ominaisuudet',
        'pricing': 'hinnat',
        'support': 'tuki',
        'blog': 'blogi',
        'resources': 'resurssit',
        'integrations': 'integraatiot',
        'templates': 'mallit',
        'affiliate-manager': 'kumppanuuspaallikkö',
        'glossary': 'sanasto',
        'academy': 'akatemia',
        'comparisons': 'vertailut',
        'directory': 'hakemisto',
        'reviews': 'arvostelut'
    },
    'pl': {
        'about': 'o-nas',
        'features': 'funkcje',
        'pricing': 'cennik',
        'support': 'wsparcie',
        'blog': 'blog',
        'resources': 'zasoby',
        'integrations': 'integracje',
        'templates': 'szablony',
        'affiliate-manager': 'menedzer-partnerski',
        'glossary': 'slownik',
        'academy': 'akademia',
        'comparisons': 'porownania',
        'directory': 'katalog',
        'reviews': 'recenzje'
    },
    'ro': {
        'about': 'despre',
        'features': 'caracteristici',
        'pricing': 'preturi',
        'support': 'suport',
        'blog': 'blog',
        'resources': 'resurse',
        'integrations': 'integrari',
        'templates': 'sabloane',
        'affiliate-manager': 'manager-afiliat',
        'glossary': 'glosar',
        'academy': 'academie',
        'comparisons': 'comparatii',
        'directory': 'director',
        'reviews': 'recenzii'
    },
    'hu': {
        'about': 'rolunk',
        'features': 'funkciok',
        'pricing': 'arak',
        'support': 'tamogatas',
        'blog': 'blog',
        'resources': 'forrasok',
        'integrations': 'integraciok',
        'templates': 'sablonok',
        'affiliate-manager': 'partner-menedzser',
        'glossary': 'szotar',
        'academy': 'akademia',
        'comparisons': 'osszehasonlitasok',
        'directory': 'konyvtar',
        'reviews': 'velemenyek'
    },
    'tr': {
        'about': 'hakkinda',
        'features': 'ozellikler',
        'pricing': 'fiyatlar',
        'support': 'destek',
        'blog': 'blog',
        'resources': 'kaynaklar',
        'integrations': 'entegrasyonlar',
        'templates': 'sablonlar',
        'affiliate-manager': 'ortaklik-yoneticisi',
        'glossary': 'sozluk',
        'academy': 'akademi',
        'comparisons': 'karsilastirmalar',
        'directory': 'dizin',
        'reviews': 'incelemeler'
    },
    'ar': {
        'about': 'حول',
        'features': 'مميزات',
        'pricing': 'الأسعار',
        'support': 'الدعم',
        'blog': 'مدونة',
        'resources': 'موارد',
        'integrations': 'تكاملات',
        'templates': 'قوالب',
        'affiliate-manager': 'مدير-الشركاء',
        'glossary': 'قاموس',
        'academy': 'الأكاديمية',
        'comparisons': 'مقارنات',
        'directory': 'دليل',
        'reviews': 'مراجعات'
    },
    'ja': {
        'about': '私たちについて',
        'features': '機能',
        'pricing': '価格',
        'support': 'サポート',
        'blog': 'ブログ',
        'resources': 'リソース',
        'integrations': '統合',
        'templates': 'テンプレート',
        'affiliate-manager': 'アフィリエイトマネージャー',
        'glossary': '用語集',
        'academy': 'アカデミー',
        'comparisons': '比較',
        'directory': 'ディレクトリ',
        'reviews': 'レビュー'
    },
    'ko': {
        'about': '소개',
        'features': '기능',
        'pricing': '가격',
        'support': '지원',
        'blog': '블로그',
        'resources': '자료',
        'integrations': '통합',
        'templates': '템플릿',
        'affiliate-manager': '제휴-매니저',
        'glossary': '용어집',
        'academy': '아카데미',
        'comparisons': '비교',
        'directory': '디렉토리',
        'reviews': '리뷰'
    },
    'zh': {
        'about': '关于',
        'features': '功能',
        'pricing': '价格',
        'support': '支持',
        'blog': '博客',
        'resources': '资源',
        'integrations': '集成',
        'templates': '模板',
        'affiliate-manager': '联盟经理',
        'glossary': '词汇表',
        'academy': '学院',
        'comparisons': '比较',
        'directory': '目录',
        'reviews': '评论'
    },
    'vi': {
        'about': 've-chung-toi',
        'features': 'tinh-nang',
        'pricing': 'gia',
        'support': 'ho-tro',
        'blog': 'blog',
        'resources': 'tai-nguyen',
        'integrations': 'tich-hop',
        'templates': 'mau',
        'affiliate-manager': 'quan-ly-doi-tac',
        'glossary': 'tu-dien',
        'academy': 'hoc-vien',
        'comparisons': 'so-sanh',
        'directory': 'danh-ba',
        'reviews': 'danh-gia'
    }
}


def extract_front_matter(file_path: Path) -> Tuple[Dict, str]:
    """Extract TOML front matter from markdown file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Look for front matter between +++ delimiters
    match = re.match(r'^\+\+\+\s*\n*(.*?)\n*\+\+\+\s*\n*', content, re.DOTALL)
    if match:
        front_matter_text = match.group(1)
        try:
            front_matter = toml.loads(front_matter_text)
            remaining_content = content[match.end():]
            return front_matter, remaining_content
        except toml.TomlDecodeError as e:
            print(f"Error parsing front matter in {file_path}: {e}")
            return {}, content
    return {}, content


def update_front_matter(file_path: Path, updated_front_matter: Dict, remaining_content: str):
    """Update TOML front matter in markdown file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('+++\n')
        f.write(toml.dumps(updated_front_matter))
        f.write('+++\n')
        f.write(remaining_content)


def get_translated_path(language: str, directory_name: str) -> str:
    """Get the translated directory name for a given language and directory"""
    if language == 'en':
        return directory_name
    
    translations = DIRECTORY_TRANSLATIONS.get(language, {})
    return translations.get(directory_name, directory_name)


def get_directory_url_path(directory_path: Path, language: str) -> Optional[str]:
    """Get the URL path for a directory by checking its _index.md file or predicting from directory name"""
    index_file = directory_path / '_index.md'
    
    if index_file.exists():
        front_matter, _ = extract_front_matter(index_file)
        if 'url' in front_matter:
            return front_matter['url'].rstrip('/')
    
    # Predict from directory name
    dir_name = directory_path.name
    translated_name = get_translated_path(language, dir_name)
    return f"/{translated_name}"


def ensure_trailing_slash(url: str) -> str:
    """Ensure URL ends with a trailing slash"""
    if not url.endswith('/'):
        return url + '/'
    return url


def process_directory(lang_dir: Path, directory: Path, stats: Dict, dry_run: bool = False, verbose: bool = False):
    """Process all files in a directory and fix their URLs"""
    language = lang_dir.name
    relative_path = directory.relative_to(lang_dir)
    
    # Get the expected base URL for this directory
    base_url = get_directory_url_path(directory, language)
    
    if not base_url:
        return
    
    if verbose or dry_run:
        print(f"\nProcessing {language}/{relative_path}: base URL = {base_url}")
    
    # Process all .md files in the directory
    for file_path in directory.glob('*.md'):
        if file_path.name == '_index.md':
            # Ensure _index.md has the correct URL
            front_matter, remaining_content = extract_front_matter(file_path)
            
            if 'url' not in front_matter:
                # Add the URL if missing
                front_matter['url'] = ensure_trailing_slash(base_url)
                if not dry_run:
                    update_front_matter(file_path, front_matter, remaining_content)
                if verbose or dry_run:
                    content_dir = get_content_dir()
                    rel_file_path = file_path.relative_to(content_dir.parent)
                    print(f"  {'[DRY-RUN] Would add' if dry_run else 'Added'} URL to {rel_file_path}")
                    print(f"    New URL: {front_matter['url']}")
                stats['urls_added'] += 1
            else:
                # Check if URL needs fixing
                current_url = front_matter.get('url', '')
                expected_url = ensure_trailing_slash(base_url)
                
                if current_url != expected_url and not current_url.startswith(base_url):
                    # Only update if the URL doesn't match the expected base
                    if verbose:
                        print(f"  _index.md URL is custom, keeping: {current_url}")
        else:
            # Process regular files
            front_matter, remaining_content = extract_front_matter(file_path)
            
            if 'url' in front_matter:
                current_url = front_matter['url']
                
                # Extract the filename part from the current URL
                url_parts = current_url.rstrip('/').split('/')
                if url_parts:
                    filename_part = url_parts[-1]
                    
                    # Build the correct URL with the base path
                    expected_url = f"{base_url}/{filename_part}/"
                    
                    if current_url != expected_url:
                        front_matter['url'] = expected_url
                        if not dry_run:
                            update_front_matter(file_path, front_matter, remaining_content)
                        if verbose or dry_run:
                            content_dir = get_content_dir()
                            rel_file_path = file_path.relative_to(content_dir.parent)
                            print(f"  {'[DRY-RUN] Would fix' if dry_run else 'Fixed'} {rel_file_path}")
                            print(f"    Current URL: {current_url}")
                            print(f"    New URL:     {expected_url}")
                        stats['urls_fixed'] += 1
                    else:
                        stats['urls_correct'] += 1
            else:
                # Add URL if missing
                # Use the filename without extension as the URL slug
                filename_slug = file_path.stem
                new_url = f"{base_url}/{filename_slug}/"
                front_matter['url'] = new_url
                if not dry_run:
                    update_front_matter(file_path, front_matter, remaining_content)
                if verbose or dry_run:
                    content_dir = get_content_dir()
                    rel_file_path = file_path.relative_to(content_dir.parent)
                    print(f"  {'[DRY-RUN] Would add' if dry_run else 'Added'} URL to {rel_file_path}")
                    print(f"    New URL: {new_url}")
                stats['urls_added'] += 1


def main():
    """Main function to process all translation directories"""
    parser = argparse.ArgumentParser(
        description='Sync translation URLs with directory structure in Hugo content'
    )
    parser.add_argument(
        '--hugo-root',
        help='Path to Hugo root directory (default: auto-detect)',
        default=None
    )
    parser.add_argument(
        '--languages',
        help='Comma-separated list of languages to process (default: all except en)',
        default=None
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without modifying files'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed output'
    )
    
    args = parser.parse_args()
    
    # Get content directory
    content_dir = get_content_dir(args.hugo_root)
    
    if not content_dir.exists():
        print(f"Error: Content directory not found at {content_dir}")
        sys.exit(1)
    
    print(f"Processing content directory: {content_dir}")
    
    stats = {
        'urls_fixed': 0,
        'urls_added': 0,
        'urls_correct': 0,
        'directories_processed': 0
    }
    
    # Determine which languages to process
    if args.languages:
        languages_to_process = args.languages.split(',')
    else:
        languages_to_process = None
    
    # Process each language directory
    for lang_dir in content_dir.iterdir():
        if lang_dir.is_dir() and lang_dir.name != 'en':  # Skip English
            # Check if we should process this language
            if languages_to_process and lang_dir.name not in languages_to_process:
                continue
                
            print(f"\n=== Processing language: {lang_dir.name} ===")
            
            # Process subdirectories
            for subdir in lang_dir.iterdir():
                if subdir.is_dir() and not subdir.name.startswith('.'):
                    process_directory(lang_dir, subdir, stats, dry_run=args.dry_run, verbose=args.verbose)
                    stats['directories_processed'] += 1
                    
                    # Also process nested subdirectories (e.g., about/team)
                    for nested_dir in subdir.iterdir():
                        if nested_dir.is_dir() and not nested_dir.name.startswith('.'):
                            process_directory(lang_dir, nested_dir, stats, dry_run=args.dry_run, verbose=args.verbose)
                            stats['directories_processed'] += 1
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Directories processed: {stats['directories_processed']}")
    print(f"URLs fixed: {stats['urls_fixed']}")
    print(f"URLs added: {stats['urls_added']}")
    print(f"URLs already correct: {stats['urls_correct']}")
    
    if args.dry_run:
        print("\nDRY RUN - No files were modified")
    else:
        print("\nTranslation URL sync complete.")


if __name__ == "__main__":
    main()