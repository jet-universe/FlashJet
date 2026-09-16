from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
project = "FlashJet"
author = "Sitian Qian, Chirayu Gupta, Alexandre De Moor, OpenAI Codex, Anthropic Claude"
copyright = "2026, FlashJet contributors"
from flashjet import __version__
release = __version__
extensions = ["myst_parser", "sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode"]
autodoc_mock_imports = ["torch", "triton"]
autodoc_typehints = "none"
html_theme = "sphinx_rtd_theme"
html_baseurl = "https://jet-universe.github.io/FlashJet/"
exclude_patterns = ["_build", "requirements.txt"]
myst_heading_anchors = 3
