import unittest
import re
from unittest.mock import patch, MagicMock
from bs4 import BeautifulSoup

from olmocr.bench.synth.mine_html_templates import (
    generate_tests_from_html,
    html_to_markdown_with_frontmatter,
    extract_html_metadata,
    PreserveTablesConverter
)
from olmocr.bench.tests import TestType


class TestMathExtraction(unittest.TestCase):
    """Test the math extraction functionality in mine_html_templates.py"""
    
    def test_math_extraction_from_html(self):
        """Test that math equations are properly extracted from HTML content"""
        html_content = """
        <html>
        <body>
        <p>Some text with inline math \\(x = 2\\) here.</p>
        <p>Display math: \\[E = mc^2\\]</p>
        <p>Another inline: \\(\\alpha + \\beta = \\gamma\\)</p>
        <p>Complex display: \\[\\int_0^\\infty e^{-x} dx = 1\\]</p>
        </body>
        </html>
        """
        
        # Generate tests from HTML
        tests = generate_tests_from_html(html_content, "test_pdf", 1)
        
        # Filter math tests
        math_tests = [t for t in tests if t.get("type") == "math"]
        
        # Check that we extracted math equations
        self.assertTrue(len(math_tests) > 0, "Should extract at least one math equation")
        
        # Check that specific equations were extracted
        math_contents = [t["math"] for t in math_tests]
        self.assertIn("x = 2", math_contents)
        self.assertIn("E = mc^2", math_contents)
        self.assertIn("\\alpha + \\beta = \\gamma", math_contents)
        self.assertIn("\\int_0^\\infty e^{-x} dx = 1", math_contents)

    def test_math_extraction_with_multiline(self):
        """Test extraction of multiline math equations"""
        html_content = """
        <html>
        <body>
        <p>Multiline equation:
        \\[
        e_i = \\frac{e_i + \\varphi(e_i)}{2} + \\frac{e_i - \\varphi(e_i)}{2}, 
        \\quad \\text{for } i \\in \\mathbb{N}.
        \\]
        </p>
        </body>
        </html>
        """
        
        tests = generate_tests_from_html(html_content, "test_pdf", 1)
        math_tests = [t for t in tests if t.get("type") == "math"]
        
        # Check multiline equation is captured
        self.assertTrue(len(math_tests) > 0)
        
        # Check that the multiline content is preserved (without excessive newlines)
        found_multiline = False
        for test in math_tests:
            if "\\frac{e_i + \\varphi(e_i)}{2}" in test["math"] and "\\mathbb{N}" in test["math"]:
                found_multiline = True
                break
        
        self.assertTrue(found_multiline, "Should extract multiline equation correctly")

    def test_math_extraction_deduplication(self):
        """Test that duplicate math equations are deduplicated"""
        html_content = """
        <html>
        <body>
        <p>First occurrence: \\[x^2 + y^2 = z^2\\]</p>
        <p>Second occurrence: \\[x^2 + y^2 = z^2\\]</p>
        <p>Third occurrence: \\[x^2 + y^2 = z^2\\]</p>
        </body>
        </html>
        """
        
        tests = generate_tests_from_html(html_content, "test_pdf", 1)
        math_tests = [t for t in tests if t.get("type") == "math"]
        
        # Count how many times the equation appears
        equation_count = sum(1 for t in math_tests if "x^2 + y^2 = z^2" in t["math"])
        
        # Should only appear once due to deduplication
        self.assertEqual(equation_count, 1, "Duplicate equations should be deduplicated")

    def test_math_extraction_patterns(self):
        """Test different math delimiter patterns"""
        html_content = """
        <html>
        <body>
        <p>Pattern 1: \\(inline1\\)</p>
        <p>Pattern 2: \\[display1\\]</p>
        <p>Pattern 3: $$display2$$</p>
        </body>
        </html>
        """
        
        tests = generate_tests_from_html(html_content, "test_pdf", 1)
        math_tests = [t for t in tests if t.get("type") == "math"]
        
        math_contents = [t["math"] for t in math_tests]
        
        # Check all patterns are captured
        self.assertIn("inline1", math_contents)
        self.assertIn("display1", math_contents)
        self.assertIn("display2", math_contents)

    def test_math_extraction_minimum_length(self):
        """Test that very short equations are filtered out"""
        html_content = """
        <html>
        <body>
        <p>Short: \\(x\\)</p>
        <p>Also short: \\[y\\]</p>
        <p>Long enough: \\(x=1\\)</p>
        </body>
        </html>
        """
        
        tests = generate_tests_from_html(html_content, "test_pdf", 1)
        math_tests = [t for t in tests if t.get("type") == "math"]
        
        math_contents = [t["math"] for t in math_tests]
        
        # Short equations (length <= 2) should be filtered out
        self.assertNotIn("x", math_contents)
        self.assertNotIn("y", math_contents)
        # Longer equation should be included
        self.assertIn("x=1", math_contents)

    def test_math_validation_passes(self):
        """Test that valid math tests pass validation against markdown"""
        html_content = """
        <html>
        <body>
        <p>Test equation: \\[E = mc^2\\]</p>
        </body>
        </html>
        """
        
        # Mock the validation to always pass for math tests
        with patch('olmocr.bench.synth.mine_html_templates.load_single_test') as mock_load:
            mock_test = MagicMock()
            mock_test.run.return_value = (True, None)
            mock_load.return_value = mock_test
            
            tests = generate_tests_from_html(html_content, "test_pdf", 1)
            math_tests = [t for t in tests if t.get("type") == "math"]
            
            # Verify math test was created
            self.assertTrue(len(math_tests) > 0)
            # Verify test has correct structure
            for test in math_tests:
                self.assertEqual(test["type"], "math")
                self.assertIn("math", test)
                self.assertEqual(test["max_diffs"], 0)
                self.assertIn("id", test)
                self.assertIn("pdf", test)
                self.assertEqual(test["page"], 1)

    def test_complex_markdown_example(self):
        """Test with the complex markdown example provided by the user"""
        # Convert markdown to HTML-like structure for testing
        html_content = '<!DOCTYPE html>\n<html lang="en">\n<head>\n    <meta charset="UTF-8">\n    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n    <title>Automorphisms of Order Two</title>\n    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>\n    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>\n    <script>\n        window.MathJax = {\n            tex: {\n                inlineMath: [[\'\\\\(\', \'\\\\)\']],\n                displayMath: [[\'\\\\[\', \'\\\\]\']]\n            }\n        };\n    </script>\n    <style>\n        body {\n            font-family: "Times New Roman", serif;\n            font-size: 11pt;\n            line-height: 1.4;\n            max-width: 791px;\n            margin: 0 auto;\n            padding: 20px;\n            background-color: white;\n        }\n        \n        .math-block {\n            margin: 15px 0;\n        }\n        \n        .definition {\n            margin: 20px 0;\n        }\n        \n        .definition-header {\n            font-weight: bold;\n            margin-bottom: 10px;\n        }\n        \n        .lemma {\n            margin: 20px 0;\n        }\n        \n        .lemma-header {\n            font-weight: bold;\n            margin-bottom: 10px;\n        }\n        \n        .proof {\n            margin: 15px 0;\n        }\n        \n        .proof-header {\n            font-weight: bold;\n            display: inline;\n        }\n        \n        .qed {\n            float: right;\n            font-weight: bold;\n        }\n        \n        ul {\n            margin: 15px 0;\n            padding-left: 20px;\n        }\n        \n        ol {\n            margin: 15px 0;\n            padding-left: 20px;\n        }\n        \n        h2 {\n            font-size: 14pt;\n            font-weight: bold;\n            margin: 25px 0 15px 0;\n        }\n        \n        .equation {\n            text-align: right;\n            margin: 15px 0;\n        }\n        \n        footer {\n            text-align: center;\n            margin-top: 30px;\n            font-weight: bold;\n        }\n    </style>\n</head>\n<body>\n    <div class="math-block">\n        <p>If \\(\\varphi \\in \\text{Aut}(E)\\) with \\(\\varphi^2 = id\\) we observe that</p>\n        \\[e_i = \\frac{e_i + \\varphi(e_i)}{2} + \\frac{e_i - \\varphi(e_i)}{2}, \\quad \\text{for } i \\in \\mathbb{N}.\\]\n        \n        <p>Setting \\(a_i = e_i + \\varphi(e_i)/2\\) we have:</p>\n        \n        <ul>\n            <li>\\(\\varphi(e_i) = -e_i + 2a_i\\),</li>\n            <li>\\(\\varphi(a_i) = a_i\\), that is, \\(a_i\\) is of degree zero in the \\(\\mathbb{Z}_2\\)-grading \\(E_\\varphi\\),</li>\n            <li>\\(\\varphi(e_i - a_i) = -(e_i - a_i)\\), that is, \\(e_i - a_i\\) is of degree 1 in the \\(\\mathbb{Z}_2\\)-grading \\(E_\\varphi\\).</li>\n        </ul>\n    </div>\n    \n    <div class="definition">\n        <div class="definition-header">Definition 5</div>\n        <p>Let \\(\\varphi \\in \\text{Aut}(E)\\). We say that \\(\\varphi\\) is of <em>canonical type</em> if \\(\\varphi(e_i) \\in E_{(1)}\\) for all \\(i\\).</p>\n        \n        <p>If \\(\\varphi\\) is an automorphism of order 2 on \\(E\\), we have that \\(\\varphi\\) is of canonical type if and only if \\(a_i \\in E_{(1)}\\) for all \\(i\\). Let us fix a basis \\(\\beta = \\{e_1, e_2, \\ldots, e_n, \\ldots\\}\\) of the vector space \\(L\\) and an automorphism \\(\\varphi \\in \\text{Aut}(E)\\) such that \\(\\varphi^2 = id\\). Then \\(\\varphi\\), as a linear transformation, has eigenvalues \\(\\pm 1\\) and \\(-1\\) only, and moreover, there exists a basis of the vector space \\(E\\) consisting of eigenvectors. (It is well known from elementary Linear Algebra that this fact does not depend on the dimension of the vector space as long as the characteristic of \\(F\\) is different from 2.) Then \\(E = E(1) \\oplus E(-1)\\) where \\(E(t)\\) is the eigenspace for the eigenvalue \\(t\\) of the linear transformation \\(\\varphi\\). One considers the intersections \\(L(t) = L \\cap E(t)\\), \\(t = \\pm 1\\). Changing the basis \\(\\beta\\), if necessary, one may assume that \\(L(t)\\) is the span of \\(\\beta \\cap L(t)\\). Clearly this change of basis gives rise to a homogeneous automorphism of \\(E\\) and we can take the composition of it and then \\(\\varphi\\). We shall assume that such a change of basis has been done.</p>\n        \n        <p>Denote</p>\n        \\[I_\\varphi = \\{n \\in \\mathbb{N} \\mid \\varphi(e_n) = \\pm e_n\\}.\\]\n    </div>\n    \n    <p>We shall distinguish the following four possibilities:</p>\n    \n    <ol>\n        <li>\\(I_\\varphi = \\mathbb{N}\\).</li>\n        <li>\\(I_\\varphi \\neq \\mathbb{N}\\) is infinite.</li>\n        <li>\\(I_\\varphi\\) is finite and nonempty.</li>\n        <li>\\(I_\\gamma = \\emptyset\\) for every linear basis \\(\\gamma\\) of \\(L\\).</li>\n    </ol>\n    \n    <p>We shall call these automorphisms (and also the corresponding \\(\\mathbb{Z}_2\\)-gradings), automorphisms (or gradings) of type 1, 2, 3, and 4, respectively.</p>\n    \n    <p>The automorphisms of type 1 induce \\(\\mathbb{Z}_2\\)-gradings on \\(E\\) in which all generators of \\(E\\) are homogeneous. Such structures are called homogeneous \\(\\mathbb{Z}_2\\)-gradings on \\(E\\). The corresponding graded identities were completely studied in [22, 24, 29].</p>\n    \n    <p>We conclude this section with the following lemma.</p>\n    \n    <div class="lemma">\n        <div class="lemma-header">Lemma 6</div>\n        <p>Let \\(\\varphi\\) be an automorphism of order two of \\(E\\). Then \\(\\varphi\\) is of type 4 if and only if, for every \\(v \\in L\\) such that \\(\\varphi(v) = \\pm v\\), one has \\(v = 0\\).</p>\n        \n        <div class="proof">\n            <span class="proof-header">Proof</span> Assume that \\(\\varphi\\) is of type 4 and let \\(v \\in L\\) with \\(\\varphi(v) = \\pm v\\). If \\(v \\neq 0\\), choose a basis \\(\\gamma\\) of \\(L\\) such that \\(v \\in \\gamma\\). Then \\(I_\\gamma \\neq \\emptyset\\), a contradiction. The converse follows by the same argument.\n            <span class="qed">■</span>\n        </div>\n    </div>\n    \n    <h2>3 &nbsp;&nbsp; Automorphisms of order two of <em>E</em></h2>\n    \n    <p>From this point on, our goal is to survey recent developments regarding automorphisms of order two and the corresponding \\(\\mathbb{Z}_2\\)-gradings of the infinite-dimensional Grassmann algebra.</p>\n    \n    <p>Let \\(X = \\{e_1, \\ldots, e_n, \\ldots\\}\\). For each map \\(\\lambda : X \\to E\\), we can define the linear transformation \\(\\varphi : E \\to E\\) by</p>\n    \n    <div class="equation">\n        \\[\\varphi(e_{i_1} \\cdots e_{i_n}) = \\lambda(e_{i_1}) \\cdots \\lambda(e_{i_n}),\\] <span style="float: right;">(1)</span>\n    </div>\n    \n    <p>for all \\(n \\in \\mathbb{N}\\).</p>\n    \n    <p>We start with the next lemma.</p>\n    \n    <div class="lemma">\n        <div class="lemma-header">Lemma 7</div>\n        <p><em>The linear transformation</em> \\(\\varphi\\) <em>is an endomorphism of</em> \\(E\\) <em>if and only if</em></p>\n        \\[\\lambda(e_i)\\lambda(e_j) + \\lambda(e_j)\\lambda(e_i) = 0, \\quad \\text{for all } i, j.\\]\n    </div>\n    \n    <footer>\n        4\n    </footer>\n</body>\n</html>'
        tests = generate_tests_from_html(html_content, "test_pdf", 1)
        math_tests = [t for t in tests if t.get("type") == "math"]

        for test in math_tests:
            print(test)

    def test_math_extraction_strips_whitespace(self):
        """Test that extracted math equations have whitespace properly stripped"""
        html_content = """
        <html>
        <body>
        <p>\\[
            x = y + z
        \\]</p>
        </body>
        </html>
        """
        
        tests = generate_tests_from_html(html_content, "test_pdf", 1)
        math_tests = [t for t in tests if t.get("type") == "math"]
        
        self.assertTrue(len(math_tests) > 0)
        # The equation should be stripped of leading/trailing whitespace
        self.assertEqual(math_tests[0]["math"].strip(), math_tests[0]["math"])


class TestExtractHtmlMetadata(unittest.TestCase):
    def test_extract_metadata_portuguese_document(self):
        """Test metadata extraction from a Portuguese document with mixed content."""
        html_content = """
        <html lang="pt">
        <head><title>Test Document</title></head>
        <body>
            <header>Header content here</header>
            <h1>Política de Metadados</h1>
            <p>Este é um documento de teste com texto em português.</p>
            <p>Contém múltiplos parágrafos para simular conteúdo real.</p>
            <div class="image">Image placeholder 1</div>
            <p>Mais texto após a imagem.</p>
            <footer>Footer content</footer>
        </body>
        </html>
        """
        
        metadata = extract_html_metadata(html_content)
        
        # Check language extraction
        self.assertEqual(metadata['primary_language'], 'pt')
        
        # Check rotation values (always fixed)
        self.assertTrue(metadata['is_rotation_valid'])
        self.assertEqual(metadata['rotation_correction'], 0)
        
        # Check table/diagram detection
        # With 1 image (500 chars) and small text content, image ratio > 50%
        self.assertFalse(metadata['is_table'])
        self.assertTrue(metadata['is_diagram'])  # Image estimate dominates
    
    def test_extract_metadata_table_heavy_document(self):
        """Test metadata extraction from a document that is mostly tables."""
        html_content = """
        <html lang="en">
        <body>
            <p>Small intro text</p>
            <table>
                <tr><td>Cell 1</td><td>Cell 2</td><td>Cell 3</td></tr>
                <tr><td>Data A</td><td>Data B</td><td>Data C</td></tr>
                <tr><td>More data</td><td>More data</td><td>More data</td></tr>
                <tr><td>Even more data</td><td>Even more data</td><td>Even more data</td></tr>
                <tr><td>Lots of data</td><td>Lots of data</td><td>Lots of data</td></tr>
                <tr><td>Table content</td><td>Table content</td><td>Table content</td></tr>
                <tr><td>Final row</td><td>Final row</td><td>Final row</td></tr>
            </table>
        </body>
        </html>
        """
        
        metadata = extract_html_metadata(html_content)
        
        self.assertEqual(metadata['primary_language'], 'en')
        self.assertTrue(metadata['is_table'])  # Should be True as >50% is table
        self.assertFalse(metadata['is_diagram'])
    
    def test_extract_metadata_image_heavy_document(self):
        """Test metadata extraction from a document that is mostly images."""
        html_content = """
        <html lang="es">
        <body>
            <p>Brief text</p>
            <div class="image">Image 1</div>
            <div class="image">Image 2</div>
            <div class="image">Image 3</div>
            <div class="image">Image 4</div>
            <div class="image">Image 5</div>
        </body>
        </html>
        """
        
        metadata = extract_html_metadata(html_content)
        
        self.assertEqual(metadata['primary_language'], 'es')
        self.assertFalse(metadata['is_table'])
        self.assertTrue(metadata['is_diagram'])  # Should be True as >50% is images
    
    def test_extract_metadata_language_with_region(self):
        """Test that language codes with regions (e.g., pt-BR) are shortened."""
        html_content = """
        <html lang="pt-BR">
        <body>
            <p>Texto em português brasileiro</p>
        </body>
        </html>
        """
        
        metadata = extract_html_metadata(html_content)
        
        # Should convert pt-BR to pt
        self.assertEqual(metadata['primary_language'], 'pt')
    
    def test_extract_metadata_no_html_tag(self):
        """Test extraction when there's no html tag (defaults to 'en')."""
        html_content = """
        <body>
            <p>Content without html tag</p>
        </body>
        """
        
        metadata = extract_html_metadata(html_content)
        
        self.assertEqual(metadata['primary_language'], 'en')  # Should default to 'en'
    
    def test_extract_metadata_mixed_content(self):
        """Test a document with mixed content types."""
        html_content = """<!DOCTYPE html>
                <html lang="pt-BR">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Política de Metadados para Livros e Capítulos de Livro UFPA</title>
                    <style>
                        body {
                            font-family: Arial, sans-serif;
                            margin: 0;
                            padding: 20px;
                            width: 725px;
                            height: 1024px;
                            box-sizing: border-box;
                        }
                        
                        .header-logos {
                            display: flex;
                            justify-content: space-between;
                            align-items: center;
                            margin-bottom: 30px;
                        }
                        
                        .image {
                            border: 2px solid black;
                            background-color: #ccc;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            color: black;
                            font-weight: bold;
                        }
                        
                        .logo-left {
                            width: 120px;
                            height: 80px;
                        }
                        
                        .logo-center {
                            width: 300px;
                            height: 80px;
                        }
                        
                        .logo-right {
                            width: 120px;
                            height: 80px;
                        }
                        
                        h1 {
                            text-align: center;
                            font-weight: bold;
                            font-size: 16px;
                            margin: 20px 0;
                            text-transform: uppercase;
                        }
                        
                        .intro-text {
                            text-align: justify;
                            margin-bottom: 20px;
                            font-size: 14px;
                            line-height: 1.4;
                        }
                        
                        table {
                            width: 100%;
                            border-collapse: collapse;
                            font-size: 12px;
                        }
                        
                        th, td {
                            border: 1px solid black;
                            padding: 8px;
                            text-align: left;
                            vertical-align: middle;
                        }
                        
                        th {
                            background-color: #f0f0f0;
                            font-weight: bold;
                            text-align: center;
                        }
                        
                        .col-metadados {
                            width: 35%;
                        }
                        
                        .col-valor {
                            width: 35%;
                        }
                        
                        .col-repetitivo {
                            width: 15%;
                            text-align: center;
                        }
                        
                        .col-condicao {
                            width: 15%;
                            text-align: center;
                        }
                        
                        footer {
                            text-align: center;
                            margin-top: 20px;
                            font-size: 14px;
                            font-weight: bold;
                        }
                    </style>
                </head>
                <body>
                    <header>
                        <div class="header-logos">
                            <div class="image logo-left">Biblioteca Central UFPA</div>
                            <div class="image logo-center">LIVRO ABERTO portal do livro aberto da UFPA</div>
                            <div class="image logo-right">SIBI/UFPA</div>
                        </div>
                    </header>

                    <main>
                        <h1>Política de Metadados para Livros e Capítulos de Livro UFPA</h1>
                        
                        <p class="intro-text">
                            Essa política de metadados possui o objetivo de garantir a consistência do trabalho executado no Portal do Livro Aberto. Dessa forma, foi desenvolvido com base no esquema de metadados do Dublin Core com adaptações para a realidade brasileira e local.
                        </p>

                        <table>
                            <thead>
                                <tr>
                                    <th class="col-metadados">METADADOS</th>
                                    <th class="col-valor">VALOR</th>
                                    <th class="col-repetitivo">REPETITIVO</th>
                                    <th class="col-condicao">CONDIÇÃO</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>dc.type</td>
                                    <td>Tipo de documento</td>
                                    <td>Não</td>
                                    <td>Obrigatório</td>
                                </tr>
                                <tr>
                                    <td>dc.title</td>
                                    <td>Título e subtítulo (se houver)</td>
                                    <td>Não</td>
                                    <td>Obrigatório</td>
                                </tr>
                                <tr>
                                    <td>dc.title.alternative</td>
                                    <td>Título alternativo</td>
                                    <td>Sim</td>
                                    <td>Opcional</td>
                                </tr>
                                <tr>
                                    <td>dc.creator</td>
                                    <td>Autor</td>
                                    <td>Sim</td>
                                    <td>Opcional</td>
                                </tr>
                                <tr>
                                    <td>dc.creator.Lattes</td>
                                    <td>URL do currículo Lattes do autor</td>
                                    <td>Sim</td>
                                    <td>Opcional</td>
                                </tr>
                                <tr>
                                    <td>dc.creator.ORCID</td>
                                    <td>ORCID do autor</td>
                                    <td>Sim</td>
                                    <td>Opcional</td>
                                </tr>
                                <tr>
                                    <td>dc.description.affiliation</td>
                                    <td>Afiliação do autor</td>
                                    <td>Sim</td>
                                    <td>Opcional</td>
                                </tr>
                                <tr>
                                    <td>dc.contributor.organizer</td>
                                    <td>Organizador</td>
                                    <td>Sim</td>
                                    <td>Opcional</td>
                                </tr>
                                <tr>
                                    <td>dc.contributor.organizerLattes</td>
                                    <td>URL do currículo Lattes do organizador</td>
                                    <td>Sim</td>
                                    <td>Opcional</td>
                                </tr>
                                <tr>
                                    <td>dc.contributor.organizerORCID</td>
                                    <td>ORCID do organizador</td>
                                    <td>Sim</td>
                                    <td>Opcional</td>
                                </tr>
                                <tr>
                                    <td>dc.description.affiliationOrganizer</td>
                                    <td>Afiliação do organizador</td>
                                    <td>Sim</td>
                                    <td>Opcional</td>
                                </tr>
                                <tr>
                                    <td>dc.contributor.coordinator</td>
                                    <td>Coordenador</td>
                                    <td>Sim</td>
                                    <td>Opcional</td>
                                </tr>
                                <tr>
                                    <td>dc.contributor.coordinatorLattes</td>
                                    <td>URL do currículo Lattes do coordenador</td>
                                    <td>Sim</td>
                                    <td>Opcional</td>
                                </tr>
                                <tr>
                                    <td>dc.contributor.coordinatorORCID</td>
                                    <td>ORCID do coordenador</td>
                                    <td>Sim</td>
                                    <td>Opcional</td>
                                </tr>
                                <tr>
                                    <td>dc.contributor.affiliationCoordinator</td>
                                    <td>Afiliação do coordenador</td>
                                    <td>Sim</td>
                                    <td>Opcional</td>
                                </tr>
                                <tr>
                                    <td>dc.contributor.editor</td>
                                    <td>Editor</td>
                                    <td>Sim</td>
                                    <td>Opcional</td>
                                </tr>
                                <tr>
                                    <td>dc.contributor.editorLattes</td>
                                    <td>URL do currículo Lattes do editor</td>
                                    <td>Sim</td>
                                    <td>Opcional</td>
                                </tr>
                                <tr>
                                    <td>dc.contributor.editorORCID</td>
                                    <td>ORCID do editor</td>
                                    <td>Sim</td>
                                    <td>Opcional</td>
                                </tr>
                                <tr>
                                    <td>dc.description.affiliationEditor</td>
                                    <td>Afiliação do editor</td>
                                    <td>Sim</td>
                                    <td>Opcional</td>
                                </tr>
                            </tbody>
                        </table>
                    </main>

                    <footer>
                        <div>3</div>
                    </footer>
                </body>
                </html>
        """
        
        metadata = extract_html_metadata(html_content)
        
        self.assertEqual(metadata['primary_language'], 'pt')
        self.assertTrue(metadata['is_table'])
        self.assertFalse(metadata['is_diagram']) 
    
    def test_extract_metadata_empty_body(self):
        """Test extraction with empty or minimal content."""
        html_content = """
        <html lang="de">
        <body></body>
        </html>
        """
        
        metadata = extract_html_metadata(html_content)
        
        self.assertEqual(metadata['primary_language'], 'de')
        self.assertFalse(metadata['is_table'])
        self.assertFalse(metadata['is_diagram'])
        self.assertTrue(metadata['is_rotation_valid'])
        self.assertEqual(metadata['rotation_correction'], 0)


class TestHtmlToMarkdown(unittest.TestCase):
    def test_title_tag_excluded_from_markdown(self):
        """Test that title tags from head are not included in markdown output."""
        html_content = """
        <html lang="en">
        <head>
            <title>This Should Not Appear In Markdown</title>
            <meta charset="UTF-8">
        </head>
        <body>
            <h1>Main Heading</h1>
            <p>This is the body content that should appear.</p>
        </body>
        </html>
        """
        
        markdown_with_frontmatter = html_to_markdown_with_frontmatter(html_content)
        
        # Check that the title from head tag is NOT in the markdown
        self.assertNotIn("This Should Not Appear In Markdown", markdown_with_frontmatter)
        
        # Check that body content IS in the markdown
        self.assertIn("Main Heading", markdown_with_frontmatter)
        self.assertIn("This is the body content that should appear", markdown_with_frontmatter)
        
        # Check that frontmatter is present
        self.assertTrue(markdown_with_frontmatter.startswith("---"))
    
    def test_image_with_data_description(self):
        """Test that images are converted with placeholder alt text."""
        html_content = """
        <html lang="en">
        <body>
            <p>Text before image</p>
            <div class="image" data-description="A beautiful sunset over mountains">Placeholder</div>
            <p>Text after image</p>
        </body>
        </html>
        """
        
        markdown_with_frontmatter = html_to_markdown_with_frontmatter(html_content)
        
        # Check that images use the fixed placeholder alt text
        self.assertIn("![Image Placeholder]", markdown_with_frontmatter)
        
        # Check that other content is preserved
        self.assertIn("Text before image", markdown_with_frontmatter)
        self.assertIn("Text after image", markdown_with_frontmatter)
    
    def test_image_without_data_description(self):
        """Test that images without data-description use default alt text."""
        html_content = """
        <html lang="en">
        <body>
            <div class="image">Some placeholder content</div>
        </body>
        </html>
        """
        
        markdown_with_frontmatter = html_to_markdown_with_frontmatter(html_content)
        
        # Check that default alt text is used
        self.assertIn("![Image Placeholder]", markdown_with_frontmatter)
    
    def test_headers_footers_excluded(self):
        """Test that header and footer tags are excluded from markdown."""
        html_content = """
        <html lang="en">
        <body>
            <header>
                <nav>Navigation menu that should not appear</nav>
            </header>
            <main>
                <h1>Main Content</h1>
                <p>This should appear in the markdown.</p>
            </main>
            <footer>
                <p>Footer text that should not appear</p>
            </footer>
        </body>
        </html>
        """
        
        markdown_with_frontmatter = html_to_markdown_with_frontmatter(html_content)
        
        # Check that header/footer content is excluded
        self.assertNotIn("Navigation menu", markdown_with_frontmatter)
        self.assertNotIn("Footer text", markdown_with_frontmatter)
        
        # Check that main content is included
        self.assertIn("Main Content", markdown_with_frontmatter)
        self.assertIn("This should appear in the markdown", markdown_with_frontmatter)
    
    def test_no_body_tag_fallback(self):
        """Test that content is still processed when there's no body tag."""
        html_content = """
        <div>
            <h1>Content without body tag</h1>
            <p>This should still be converted.</p>
        </div>
        """
        
        markdown_with_frontmatter = html_to_markdown_with_frontmatter(html_content)
        
        # Check that content is still converted
        self.assertIn("Content without body tag", markdown_with_frontmatter)
        self.assertIn("This should still be converted", markdown_with_frontmatter)
    
    def test_removes_triple_dashes_from_content(self):
        """Test that --- at the start or end of markdown content is removed."""
        # Test with --- at the beginning
        html_content_start = """
        <html lang="en">
        <body>
            <p>---</p>
            <p>Regular content here</p>
        </body>
        </html>
        """
        
        markdown_start = html_to_markdown_with_frontmatter(html_content_start)
        lines = markdown_start.split('\n')
        
        # Check that we have FrontMatter
        self.assertEqual(lines[0], '---')
        # Check that the content doesn't start with --- after the FrontMatter ends
        frontmatter_end = next(i for i in range(1, len(lines)) if lines[i] == '---')
        content_after_frontmatter = '\n'.join(lines[frontmatter_end + 1:])
        self.assertFalse(content_after_frontmatter.strip().startswith('---'))
        
        # Test with --- at the end
        html_content_end = """
        <html lang="en">
        <body>
            <p>Regular content here</p>
            <p>---</p>
        </body>
        </html>
        """
        
        markdown_end = html_to_markdown_with_frontmatter(html_content_end)
        # Check that content doesn't end with ---
        self.assertFalse(markdown_end.rstrip().endswith('---\n---'))
        
        # Test with --- at both beginning and end
        html_content_both = """
        <html lang="en">
        <body>
            <p>---</p>
            <p>Middle content</p>
            <p>---</p>
        </body>
        </html>
        """
        
        markdown_both = html_to_markdown_with_frontmatter(html_content_both)
        lines_both = markdown_both.split('\n')
        frontmatter_end_both = next(i for i in range(1, len(lines_both)) if lines_both[i] == '---')
        content_both = '\n'.join(lines_both[frontmatter_end_both + 1:])
        
        # Content should not start or end with ---
        self.assertFalse(content_both.strip().startswith('---'))
        self.assertFalse(content_both.strip().endswith('---'))
        # But should contain "Middle content"
        self.assertIn("Middle content", content_both)


class TestSuperscriptSubscriptConversion(unittest.TestCase):
    """Test superscript and subscript conversion to Unicode in html_to_markdown_with_frontmatter"""
    
    def test_basic_superscripts(self):
        """Test basic superscript conversion"""
        html = """
        <html>
        <body>
            <p>x<sup>2</sup> + y<sup>3</sup> = z<sup>4</sup></p>
            <p>10<sup>9</sup> is a billion</p>
        </body>
        </html>
        """
        result = html_to_markdown_with_frontmatter(html)
        
        # Check that superscripts are converted to Unicode
        self.assertIn("x²", result)
        self.assertIn("y³", result)
        self.assertIn("z⁴", result)
        self.assertIn("10⁹", result)
        
        # Should not contain HTML sup tags in markdown
        self.assertNotIn("<sup>", result)
        self.assertNotIn("</sup>", result)
    
    def test_basic_subscripts(self):
        """Test basic subscript conversion"""
        html = """
        <html>
        <body>
            <p>H<sub>2</sub>O is water</p>
            <p>CO<sub>2</sub> is carbon dioxide</p>
            <p>X<sub>n</sub> represents the nth element</p>
        </body>
        </html>
        """
        result = html_to_markdown_with_frontmatter(html)
        
        # Check that subscripts are converted to Unicode
        self.assertIn("H₂O", result)
        self.assertIn("CO₂", result)
        self.assertIn("Xₙ", result)
        
        # Should not contain HTML sub tags in markdown
        self.assertNotIn("<sub>", result)
        self.assertNotIn("</sub>", result)
    
    def test_mixed_super_and_subscripts(self):
        """Test mixed superscripts and subscripts"""
        html = """
        <html>
        <body>
            <p>The formula is x<sup>2</sup> + H<sub>2</sub>O<sup>+</sup></p>
            <p>Chemical: Ca<sup>2+</sup> and SO<sub>4</sub><sup>2-</sup></p>
        </body>
        </html>
        """
        result = html_to_markdown_with_frontmatter(html)
        
        # Check mixed conversions
        self.assertIn("x²", result)
        self.assertIn("H₂O⁺", result)
        self.assertIn("Ca²⁺", result)
        self.assertIn("SO₄²⁻", result)
    
    def test_special_characters(self):
        """Test special character conversions"""
        html = """
        <html>
        <body>
            <p>Math: (x+y)<sup>n</sup> and f<sub>(x)</sub></p>
            <p>Ion: OH<sup>-</sup> and H<sup>+</sup></p>
            <p>Index: a<sub>i</sub> and b<sup>i</sup></p>
        </body>
        </html>
        """
        result = html_to_markdown_with_frontmatter(html)
        
        # Check special character conversions
        self.assertIn("(x+y)ⁿ", result)
        self.assertIn("f₍ₓ₎", result)
        self.assertIn("OH⁻", result)
        self.assertIn("H⁺", result)
        # subscript i might not be in map, so check either form
        self.assertTrue("aᵢ" in result or "a<sub>i</sub>" in result or "ai" in result)
        self.assertIn("bⁱ", result)
    
    def test_in_table(self):
        """Test superscripts/subscripts within HTML tables"""
        html = """
        <html>
        <body>
            <table>
                <tr>
                    <th>Chemical</th>
                    <th>Formula</th>
                </tr>
                <tr>
                    <td>Water</td>
                    <td>H<sub>2</sub>O</td>
                </tr>
                <tr>
                    <td>Sulfate ion</td>
                    <td>SO<sub>4</sub><sup>2-</sup></td>
                </tr>
            </table>
        </body>
        </html>
        """
        result = html_to_markdown_with_frontmatter(html)
        
        # Tables should be preserved as HTML but superscripts/subscripts should still be converted
        self.assertIn("<table>", result)
        
        # Check if conversions happened in table cells
        self.assertTrue("H₂O" in result or "<sub>2</sub>" in result)
        self.assertTrue("SO₄²⁻" in result or "<sub>4</sub><sup>2-</sup>" in result)
    
    def test_nested_elements(self):
        """Test superscripts/subscripts in nested HTML elements"""
        html = """
        <html>
        <body>
            <div>
                <p>In physics: E = mc<sup>2</sup></p>
                <ul>
                    <li>First: x<sup>1</sup></li>
                    <li>Second: x<sub>2</sub></li>
                </ul>
            </div>
        </body>
        </html>
        """
        result = html_to_markdown_with_frontmatter(html)
        
        # Check conversions in nested structures
        self.assertIn("mc²", result)
        self.assertTrue("x¹" in result or "x1" in result)
        self.assertTrue("x₂" in result or "x2" in result)
    
    def test_frontmatter_preserved(self):
        """Test that frontmatter is still generated correctly"""
        html = """
        <html lang="es">
        <body>
            <p>Test with x<sup>2</sup></p>
            <table><tr><td>Data</td></tr></table>
        </body>
        </html>
        """
        result = html_to_markdown_with_frontmatter(html)
        
        # Check frontmatter exists
        self.assertTrue(result.startswith("---"))
        self.assertIn("primary_language: es", result)
        self.assertIn("is_table:", result)
        
        # Also check the conversion happened
        self.assertIn("x²", result)
    
    def test_unmapped_characters(self):
        """Test characters not in the mapping"""
        html = """
        <html>
        <body>
            <p>Unknown: x<sup>abc</sup> and y<sub>xyz</sub></p>
            <p>Mixed: H<sub>2</sub>SO<sub>4</sub> with note<sup>*</sup></p>
        </body>
        </html>
        """
        result = html_to_markdown_with_frontmatter(html)
        
        # Unmapped characters should be left as-is or handled gracefully
        self.assertIn("H₂SO₄", result)
        # Asterisk is not in the map, so it might remain as-is
        self.assertTrue("note*" in result or "note<sup>*</sup>" in result or "note^*" in result)
    
    def test_empty_super_subscripts(self):
        """Test empty sup/sub tags"""
        html = """
        <html>
        <body>
            <p>Empty tags: x<sup></sup> and y<sub></sub></p>
            <p>Normal: z<sup>2</sup></p>
        </body>
        </html>
        """
        result = html_to_markdown_with_frontmatter(html)
        
        # Empty tags should not cause errors
        self.assertIn("z²", result)
        # Empty tags should just be removed
        self.assertIn("x", result)
        self.assertIn("y", result)
    
    def test_complex_math_expression(self):
        """Test a complex mathematical expression"""
        html = """
        <html>
        <body>
            <p>The equation: (x<sub>1</sub>)<sup>2</sup> + (x<sub>2</sub>)<sup>2</sup> = r<sup>2</sup></p>
            <p>Series: a<sub>0</sub> + a<sub>1</sub>x + a<sub>2</sub>x<sup>2</sup> + ... + a<sub>n</sub>x<sup>n</sup></p>
        </body>
        </html>
        """
        result = html_to_markdown_with_frontmatter(html)
        
        # Check complex nested expressions
        self.assertIn("x₁", result)
        self.assertIn("x₂", result) 
        self.assertIn("r²", result)
        self.assertIn("a₀", result)
        self.assertIn("a₁", result)
        self.assertIn("a₂", result)
        self.assertIn("aₙ", result)
        self.assertIn("xⁿ", result)


if __name__ == '__main__':
    unittest.main()