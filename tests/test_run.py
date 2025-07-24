import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, message="builtin type .* has no __module__ attribute")

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import pickle
import pytest
from unittest.mock import patch, MagicMock, mock_open
import run

class TestRunPy:
    @patch('run.fitz.open')
    def test_load_pdf_chunks(self, mock_fitz_open):
        mock_doc = MagicMock()
        mock_doc.__iter__.return_value = [MagicMock(get_text=MagicMock(return_value='Page 1 text')), MagicMock(get_text=MagicMock(return_value='Page 2 text'))]
        mock_fitz_open.return_value = mock_doc
        chunks = run.load_pdf_chunks('dummy.pdf', chunk_size=10, chunk_overlap=0)
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    @patch('run.FAISS')
    @patch('run.HuggingFaceEmbeddings')
    def test_build_or_load_index_builds(self, mock_hfemb, mock_faiss):
        # Simulate no index files and missing PDF
        with patch('os.path.exists', return_value=False):
            with patch('run.load_pdf_chunks', return_value=['chunk1', 'chunk2']):
                mock_faiss.from_texts.return_value = MagicMock(save_local=MagicMock())
                mock_hfemb.return_value = MagicMock()
                with patch('builtins.open', mock_open()):
                    with pytest.raises(SystemExit):
                        run.build_or_load_index()

    @patch('run.FAISS')
    @patch('run.HuggingFaceEmbeddings')
    def test_build_or_load_index_loads(self, mock_hfemb, mock_faiss):
        # Simulate index files exist
        with patch('os.path.exists', side_effect=[True, True]):
            mock_vectorstore = MagicMock()
            mock_faiss.load_local.return_value = mock_vectorstore
            mock_hfemb.return_value = MagicMock()
            fake_chunks = ['chunk1', 'chunk2']
            with patch('builtins.open', mock_open(read_data=pickle.dumps(fake_chunks))):
                with patch('pickle.load', return_value=fake_chunks):
                    vectorstore, chunks = run.build_or_load_index()
                    assert vectorstore == mock_vectorstore
                    assert chunks == fake_chunks

    def test_build_context(self):
        chunks = ["A", "B"]
        context = run.build_context(chunks)
        assert '- "A"' in context
        assert '- "B"' in context

    @patch('run.ChatOpenAI')
    def test_ask_llm(self, mock_chat_openai):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Short answer.")
        mock_chat_openai.return_value = mock_llm
        context = "- 'chunk'"
        question = "What?"
        result = run.ask_llm(context, question)
        assert result == "Short answer."

    def test_retrieve_chunks(self):
        mock_vectorstore = MagicMock()
        mock_vectorstore.similarity_search_with_score.return_value = [
            (MagicMock(page_content='chunk1'), 0.1),
            (MagicMock(page_content='chunk2'), 0.2)
        ]
        result = run.retrieve_chunks(mock_vectorstore, 'query', top_n=2)
        assert result == [('chunk1', 0.1), ('chunk2', 0.2)]

    @patch('builtins.input', return_value='When did the EDSA People Power Revolution happen?')
    @patch('run.build_or_load_index')
    @patch('run.retrieve_chunks')
    @patch('run.ask_llm')
    def test_main_flow(self, mock_ask_llm, mock_retrieve_chunks, mock_build_index, mock_input):
        mock_vectorstore = MagicMock()
        mock_build_index.return_value = (mock_vectorstore, ['chunk1', 'chunk2'])
        mock_retrieve_chunks.return_value = [
            ('The EDSA People Power Revolution occurred in February 1986...', 0.1),
            ('It led to the ousting of President Ferdinand Marcos...', 0.2)
        ]
        mock_ask_llm.return_value = "The EDSA People Power Revolution happened in February 1986 and marked the end of Marcos' dictatorship in the Philippines."
        with patch('builtins.print') as mock_print:
            run.main()
            mock_print.assert_any_call('"The EDSA People Power Revolution happened in February 1986 and marked the end of Marcos\' dictatorship in the Philippines."')

    @patch('builtins.input', return_value='Who is José Rizal and why is he important?')
    @patch('run.build_or_load_index')
    @patch('run.retrieve_chunks')
    @patch('run.ask_llm')
    def test_query_jose_rizal(self, mock_ask_llm, mock_retrieve_chunks, mock_build_index, mock_input):
        mock_vectorstore = MagicMock()
        mock_build_index.return_value = (mock_vectorstore, ['chunk1', 'chunk2'])
        mock_retrieve_chunks.return_value = [
            ('José Rizal was a Filipino nationalist and writer...', 0.1),
            ('He inspired the Philippine revolution...', 0.2)
        ]
        mock_ask_llm.return_value = "José Rizal was a Filipino nationalist and writer whose works inspired the Philippine revolution against Spanish rule."
        with patch('builtins.print') as mock_print:
            run.main()
            mock_print.assert_any_call('"José Rizal was a Filipino nationalist and writer whose works inspired the Philippine revolution against Spanish rule."')

    @patch('builtins.input', return_value='Tell me about the Spanish colonization of the Philippines.')
    @patch('run.build_or_load_index')
    @patch('run.retrieve_chunks')
    @patch('run.ask_llm')
    def test_query_spanish_colonization(self, mock_ask_llm, mock_retrieve_chunks, mock_build_index, mock_input):
        mock_vectorstore = MagicMock()
        mock_build_index.return_value = (mock_vectorstore, ['chunk1', 'chunk2'])
        mock_retrieve_chunks.return_value = [
            ('The Spanish colonization began in 1565...', 0.1),
            ('It lasted for more than 300 years...', 0.2)
        ]
        mock_ask_llm.return_value = "The Spanish colonization of the Philippines began in 1565 and lasted for over 300 years, significantly shaping the country’s culture and history."
        with patch('builtins.print') as mock_print:
            run.main()
            mock_print.assert_any_call('"The Spanish colonization of the Philippines began in 1565 and lasted for over 300 years, significantly shaping the country’s culture and history."')

    @patch('builtins.input', return_value='What is the significance of June 12, 1898 in Philippine history?')
    @patch('run.build_or_load_index')
    @patch('run.retrieve_chunks')
    @patch('run.ask_llm')
    def test_query_independence_day(self, mock_ask_llm, mock_retrieve_chunks, mock_build_index, mock_input):
        mock_vectorstore = MagicMock()
        mock_build_index.return_value = (mock_vectorstore, ['chunk1', 'chunk2'])
        mock_retrieve_chunks.return_value = [
            ('June 12, 1898 is the date of Philippine independence...', 0.1),
            ('It marked the declaration of independence from Spain...', 0.2)
        ]
        mock_ask_llm.return_value = "June 12, 1898 is significant as the day the Philippines declared independence from Spain."
        with patch('builtins.print') as mock_print:
            run.main()
            mock_print.assert_any_call('"June 12, 1898 is significant as the day the Philippines declared independence from Spain."')

    @patch('builtins.input', return_value='')
    @patch('run.build_or_load_index')
    @patch('run.retrieve_chunks')
    @patch('run.ask_llm')
    def test_empty_query(self, mock_ask_llm, mock_retrieve_chunks, mock_build_index, mock_input):
        mock_vectorstore = MagicMock()
        mock_build_index.return_value = (mock_vectorstore, ['chunk1', 'chunk2'])
        mock_retrieve_chunks.return_value = []
        mock_ask_llm.return_value = "Not enough information in the context."
        with patch('builtins.print') as mock_print:
            run.main()
            mock_print.assert_any_call('"Not enough information in the context."')

    @patch('builtins.input', return_value='What is the capital of Atlantis?')
    @patch('run.build_or_load_index')
    @patch('run.retrieve_chunks')
    @patch('run.ask_llm')
    def test_query_no_relevant_context(self, mock_ask_llm, mock_retrieve_chunks, mock_build_index, mock_input):
        mock_vectorstore = MagicMock()
        mock_build_index.return_value = (mock_vectorstore, ['chunk1', 'chunk2'])
        mock_retrieve_chunks.return_value = []
        mock_ask_llm.return_value = "Not enough information in the context."
        with patch('builtins.print') as mock_print:
            run.main()
            mock_print.assert_any_call('"Not enough information in the context."')

    @patch('builtins.input', return_value='Who was the first President of the Philippines?')
    @patch('run.build_or_load_index')
    @patch('run.retrieve_chunks')
    @patch('run.ask_llm')
    def test_query_first_president(self, mock_ask_llm, mock_retrieve_chunks, mock_build_index, mock_input):
        mock_vectorstore = MagicMock()
        mock_build_index.return_value = (mock_vectorstore, ['chunk1', 'chunk2'])
        mock_retrieve_chunks.return_value = [
            ('Emilio Aguinaldo was the first President of the Philippines...', 0.1),
        ]
        mock_ask_llm.return_value = "Emilio Aguinaldo was the first President of the Philippines."
        with patch('builtins.print') as mock_print:
            run.main()
            mock_print.assert_any_call('"Emilio Aguinaldo was the first President of the Philippines."')

    @patch('builtins.input', return_value='What happened in 1941 in the Philippines?')
    @patch('run.build_or_load_index')
    @patch('run.retrieve_chunks')
    @patch('run.ask_llm')
    def test_query_ww2(self, mock_ask_llm, mock_retrieve_chunks, mock_build_index, mock_input):
        mock_vectorstore = MagicMock()
        mock_build_index.return_value = (mock_vectorstore, ['chunk1', 'chunk2'])
        mock_retrieve_chunks.return_value = [
            ('In 1941, the Philippines was invaded by Japan...', 0.1),
        ]
        mock_ask_llm.return_value = "In 1941, the Philippines was invaded by Japan during World War II."
        with patch('builtins.print') as mock_print:
            run.main()
            mock_print.assert_any_call('"In 1941, the Philippines was invaded by Japan during World War II."')

    @patch('builtins.input', return_value='Who is Andres Bonifacio?')
    @patch('run.build_or_load_index')
    @patch('run.retrieve_chunks')
    @patch('run.ask_llm')
    def test_query_bonifacio(self, mock_ask_llm, mock_retrieve_chunks, mock_build_index, mock_input):
        mock_vectorstore = MagicMock()
        mock_build_index.return_value = (mock_vectorstore, ['chunk1', 'chunk2'])
        mock_retrieve_chunks.return_value = [
            ('Andres Bonifacio was a Filipino revolutionary leader...', 0.1),
        ]
        mock_ask_llm.return_value = "Andres Bonifacio was a Filipino revolutionary leader and founder of the Katipunan."
        with patch('builtins.print') as mock_print:
            run.main()
            mock_print.assert_any_call('"Andres Bonifacio was a Filipino revolutionary leader and founder of the Katipunan."')

    @patch('builtins.input', return_value='What is the meaning of "Bayanihan"?')
    @patch('run.build_or_load_index')
    @patch('run.retrieve_chunks')
    @patch('run.ask_llm')
    def test_query_bayanihan(self, mock_ask_llm, mock_retrieve_chunks, mock_build_index, mock_input):
        mock_vectorstore = MagicMock()
        mock_build_index.return_value = (mock_vectorstore, ['chunk1', 'chunk2'])
        mock_retrieve_chunks.return_value = [
            ('Bayanihan refers to the spirit of communal unity...', 0.1),
        ]
        mock_ask_llm.return_value = "Bayanihan refers to the spirit of communal unity and cooperation among Filipinos."
        with patch('builtins.print') as mock_print:
            run.main()
            mock_print.assert_any_call('"Bayanihan refers to the spirit of communal unity and cooperation among Filipinos."')

    @patch('builtins.input', return_value='Tell me about the Katipunan.')
    @patch('run.build_or_load_index')
    @patch('run.retrieve_chunks')
    @patch('run.ask_llm')
    def test_query_katipunan(self, mock_ask_llm, mock_retrieve_chunks, mock_build_index, mock_input):
        mock_vectorstore = MagicMock()
        mock_build_index.return_value = (mock_vectorstore, ['chunk1', 'chunk2'])
        mock_retrieve_chunks.return_value = [
            ('The Katipunan was a secret revolutionary society...', 0.1),
        ]
        mock_ask_llm.return_value = "The Katipunan was a secret revolutionary society that fought for Philippine independence from Spain."
        with patch('builtins.print') as mock_print:
            run.main()
            mock_print.assert_any_call('"The Katipunan was a secret revolutionary society that fought for Philippine independence from Spain."')

    @patch('builtins.input', return_value='Who is the current president of the Philippines?')
    @patch('run.build_or_load_index')
    @patch('run.retrieve_chunks')
    @patch('run.ask_llm')
    def test_query_current_president(self, mock_ask_llm, mock_retrieve_chunks, mock_build_index, mock_input):
        mock_vectorstore = MagicMock()
        mock_build_index.return_value = (mock_vectorstore, ['chunk1', 'chunk2'])
        mock_retrieve_chunks.return_value = []
        mock_ask_llm.return_value = "Not enough information in the context."
        with patch('builtins.print') as mock_print:
            run.main()
            mock_print.assert_any_call('"Not enough information in the context."')

    @patch('builtins.input', return_value='What is the longest river in the Philippines?')
    @patch('run.build_or_load_index')
    @patch('run.retrieve_chunks')
    @patch('run.ask_llm')
    def test_query_longest_river(self, mock_ask_llm, mock_retrieve_chunks, mock_build_index, mock_input):
        mock_vectorstore = MagicMock()
        mock_build_index.return_value = (mock_vectorstore, ['chunk1', 'chunk2'])
        mock_retrieve_chunks.return_value = [
            ('The Cagayan River is the longest river in the Philippines...', 0.1),
        ]
        mock_ask_llm.return_value = "The Cagayan River is the longest river in the Philippines."
        with patch('builtins.print') as mock_print:
            run.main()
            mock_print.assert_any_call('"The Cagayan River is the longest river in the Philippines."')

    @patch('builtins.input', return_value='Who is the national hero of the Philippines?')
    @patch('run.build_or_load_index')
    @patch('run.retrieve_chunks')
    @patch('run.ask_llm')
    def test_query_national_hero(self, mock_ask_llm, mock_retrieve_chunks, mock_build_index, mock_input):
        mock_vectorstore = MagicMock()
        mock_build_index.return_value = (mock_vectorstore, ['chunk1', 'chunk2'])
        mock_retrieve_chunks.return_value = [
            ('José Rizal is considered the national hero...', 0.1),
        ]
        mock_ask_llm.return_value = "José Rizal is considered the national hero of the Philippines."
        with patch('builtins.print') as mock_print:
            run.main()
            mock_print.assert_any_call('"José Rizal is considered the national hero of the Philippines."')

    @patch('builtins.input', return_value='What is the meaning of "Luzviminda"?')
    @patch('run.build_or_load_index')
    @patch('run.retrieve_chunks')
    @patch('run.ask_llm')
    def test_query_luzviminda(self, mock_ask_llm, mock_retrieve_chunks, mock_build_index, mock_input):
        mock_vectorstore = MagicMock()
        mock_build_index.return_value = (mock_vectorstore, ['chunk1', 'chunk2'])
        mock_retrieve_chunks.return_value = [
            ('Luzviminda is a portmanteau of Luzon, Visayas, and Mindanao...', 0.1),
        ]
        mock_ask_llm.return_value = "Luzviminda is a portmanteau of Luzon, Visayas, and Mindanao, the three main island groups of the Philippines."
        with patch('builtins.print') as mock_print:
            run.main()
            mock_print.assert_any_call('"Luzviminda is a portmanteau of Luzon, Visayas, and Mindanao, the three main island groups of the Philippines."') 