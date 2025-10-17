#!/usr/bin/env python3
"""
Demo script for testing PDF processing with chunking
Usage: python demo_pdf_processing.py <path_to_pdf> [options]
"""

import sys
import asyncio
import argparse
from pathlib import Path

# Add current directory to path to import modules
# This script is in src/api/test/, so we need to add parent directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.document_service import document_service, DocumentService
from services.document_manager import DocumentManager, ChunkMetadata
from config import settings


async def demo_pdf_processing(pdf_path: str, model_identifier: str = None, upload_dir: str = 'demo_uploads'):
    """
    Demo function to test PDF processing

    Args:
        pdf_path: Path to the PDF file to process
        model_identifier: Model name/path or serving name to use for tokenizer (optional)
    """
    print("=" * 80)
    print("PDF Processing Demo with Chunking")
    print("=" * 80)
    print()

    # Convert to Path object
    file_path = Path(pdf_path)

    # Validate file exists
    if not file_path.exists():
        print(f"❌ Error: File not found: {pdf_path}")
        return

    # Validate file extension
    if file_path.suffix.lower() != '.pdf':
        print(f"❌ Error: Not a PDF file: {pdf_path}")
        print("   Please provide a PDF file (.pdf extension)")
        return

    print(f"📄 Processing PDF: {file_path.name}")
    print(f"📁 Full path: {file_path.absolute()}")
    print(f"📦 File size: {file_path.stat().st_size / 1024:.2f} KB")

    # Get tokenizer for specified model or auto-detect
    if model_identifier:
        tokenizer_name = settings.get_tokenizer_for_model(model_identifier)
        if tokenizer_name:
            print(f"🔧 Using tokenizer from model: {model_identifier}")
            print(f"   Tokenizer: {tokenizer_name}")
        else:
            print(f"⚠️  Model '{model_identifier}' has no tokenizer configured")
            tokenizer_name = None
    else:
        tokenizer_name = settings.get_tokenizer_for_model()
        if tokenizer_name:
            print(f"🔧 Auto-detected tokenizer: {tokenizer_name}")
        else:
            print(f"ℹ️  No tokenizer configured (token_count will be None)")

    print()

    try:
        # Step 1: Process the PDF document
        print("Step 1: Extracting text and creating chunks...")
        print("-" * 80)

        # Create document service with specified tokenizer
        service = DocumentService(
            chunk_size=settings.documents.chunk_size,
            chunk_overlap=settings.documents.chunk_overlap,
            tokenizer=tokenizer_name
        )

        processed_doc = await service.process_document(file_path)

        print(f"✅ PDF processed successfully!")
        print()
        print(f"📊 Document Statistics:")
        print(f"   - Total pages: {processed_doc.total_pages}")
        print(f"   - Total chunks: {processed_doc.total_chunks}")
        print(f"   - Full text length: {len(processed_doc.full_text):,} characters")
        print(f"   - Average chars per page: {len(processed_doc.full_text) // processed_doc.total_pages:,}")
        print(f"   - Average chars per chunk: {len(processed_doc.full_text) // processed_doc.total_chunks:,}")
        print()

        # Step 2: Show chunk information
        print("Step 2: Analyzing chunks...")
        print("-" * 80)

        num_preview = min(5, len(processed_doc.chunks))
        print(f"\nShowing first {num_preview} of {processed_doc.total_chunks} chunks:\n")

        for i, chunk in enumerate(processed_doc.chunks[:num_preview]):
            print(f"Chunk #{chunk.chunk_id}:")
            print(f"   Pages: {chunk.page_numbers}")
            print(f"   Characters: {chunk.char_count}")
            if chunk.token_count is not None:
                print(f"   Tokens: {chunk.token_count} (ratio: {chunk.token_count/chunk.char_count:.3f})")
                print(f"   Tokenizer: {processed_doc.tokenizer}")
            else:
                print(f"   Tokens: None (no tokenizer)")
            print(f"   Preview: {chunk.content[:150].replace(chr(10), ' ')}...")
            print()

        if processed_doc.total_chunks > num_preview:
            print(f"... and {processed_doc.total_chunks - num_preview} more chunks")

        print()

        # Step 3: Show full text preview
        print("Step 3: Full text preview...")
        print("-" * 80)
        print()

        preview_length = min(500, len(processed_doc.full_text))
        print(processed_doc.full_text[:preview_length])

        if len(processed_doc.full_text) > preview_length:
            print(f"\n... (showing first {preview_length} of {len(processed_doc.full_text)} characters)")

        print()

        # Step 4: Test document manager integration
        print("Step 4: Testing DocumentManager integration...")
        print("-" * 80)

        # Create a temporary document manager for demo
        demo_manager = DocumentManager(upload_dir=upload_dir)

        # Check if document already exists and handle force option
        existing_doc_id = None
        # Always check if document with same filename already exists
        for doc_meta in demo_manager.list_documents():
            if doc_meta['filename'] == file_path.name:
                existing_doc_id = doc_meta['doc_id']
                break

        if existing_doc_id:
            print(f"🔄 Document '{file_path.name}' already exists with ID: {existing_doc_id}")
            print("   Removing existing document to process latest version...")
            demo_manager.delete_document(existing_doc_id)
            print(f"✅ Existing document removed")
            print()

        # Convert DocumentChunk to ChunkMetadata
        chunks = [
            ChunkMetadata(**chunk.to_dict())
            for chunk in processed_doc.chunks
        ]

        # Add document to managercc
        doc_id = demo_manager.add_document(
            filename=file_path.name,
            file_size=file_path.stat().st_size,
            file_path=file_path,
            full_text=processed_doc.full_text,
            chunks=chunks,
            total_pages=processed_doc.total_pages,
            tokenizer=processed_doc.tokenizer
        )

        print(f"✅ Document added to manager with ID: {doc_id}")
        print()

        # Retrieve document
        doc = demo_manager.get_document(doc_id)
        if doc:
            print(f"📋 Document metadata:")
            doc_dict = doc.to_dict(include_preview=True, include_chunks=True)
            for key, value in doc_dict.items():
                if key == 'chunks':
                    print(f"   - {key}: [{len(value)} chunks]")
                elif key == 'content_preview':
                    print(f"   - {key}: {value[:80]}...")
                else:
                    print(f"   - {key}: {value}")

        print()

        # Test chunk retrieval
        print("Step 5: Testing chunk retrieval...")
        print("-" * 80)

        first_chunk = demo_manager.get_document_chunk(doc_id, 0)
        if first_chunk:
            print(f"\n✅ Retrieved chunk #0:")
            print(f"   - Pages: {first_chunk.page_numbers}")
            print(f"   - Characters: {first_chunk.char_count}")
            if first_chunk.token_count is not None:
                print(f"   - Tokens: {first_chunk.token_count}")
                # Get tokenizer from document level
                doc = demo_manager.get_document(doc_id)
                if doc and doc.tokenizer:
                    print(f"   - Tokenizer: {doc.tokenizer}")
            print(f"   - Content preview: {first_chunk.content[:100].replace(chr(10), ' ')}...")

        all_chunks = demo_manager.get_document_chunks(doc_id)
        if all_chunks:
            print(f"\n✅ Retrieved all {len(all_chunks)} chunks successfully")

        # Show token statistics if available
        if processed_doc.chunks and processed_doc.chunks[0].token_count is not None:
            total_tokens = sum(chunk.token_count for chunk in processed_doc.chunks if chunk.token_count)
            print(f"\n📊 Token Statistics:")
            print(f"   - Total tokens: {total_tokens:,}")
            print(f"   - Avg tokens per chunk: {total_tokens // len(processed_doc.chunks)}")
            print(f"   - Avg token/char ratio: {total_tokens / len(processed_doc.full_text):.3f}")

        print()
        print("=" * 80)
        print("✅ Demo completed successfully!")
        print("=" * 80)
        print()
        print("Summary:")
        print(f"  📄 File: {file_path.name}")
        print(f"  📊 Pages: {processed_doc.total_pages}")
        print(f"  🔢 Chunks: {processed_doc.total_chunks}")
        print(f"  📝 Characters: {len(processed_doc.full_text):,}")
        if processed_doc.chunks and processed_doc.chunks[0].token_count is not None:
            total_tokens = sum(chunk.token_count for chunk in processed_doc.chunks if chunk.token_count)
            print(f"  🎯 Tokens: {total_tokens:,}")
        print()

    except Exception as e:
        print(f"\n❌ Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()


async def demo_with_custom_chunk_size(
    pdf_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    model_identifier: str = None,
    upload_dir: str = 'demo_uploads'
):
    """
    Demo function with custom chunk size

    Args:
        pdf_path: Path to the PDF file
        chunk_size: Custom chunk size in characters
        chunk_overlap: Custom overlap in characters
        model_identifier: Model name/path or serving name to use for tokenizer (optional)
    """
    print("=" * 80)
    print(f"PDF Processing Demo with Custom Chunk Size")
    print(f"Chunk Size: {chunk_size}, Overlap: {chunk_overlap}")
    print("=" * 80)
    print()

    file_path = Path(pdf_path)

    if not file_path.exists():
        print(f"❌ Error: File not found: {pdf_path}")
        return

    print(f"📄 Processing: {file_path.name}")

    # Get tokenizer
    tokenizer_name = settings.get_tokenizer_for_model(model_identifier) if model_identifier else settings.get_tokenizer_for_model()
    if tokenizer_name:
        print(f"🔧 Using tokenizer: {tokenizer_name}")
    print()

    # Create custom document service
    custom_service = DocumentService(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        tokenizer=tokenizer_name
    )

    try:
        processed_doc = await custom_service.process_document(file_path)

        print(f"✅ PDF processed with custom settings!")
        print()
        print(f"📊 Statistics:")
        print(f"   - Total pages: {processed_doc.total_pages}")
        print(f"   - Total chunks: {processed_doc.total_chunks}")
        print(f"   - Full text: {len(processed_doc.full_text):,} chars")
        print(f"   - Average chunk size: {len(processed_doc.full_text) // processed_doc.total_chunks} chars")
        print()

        # Show chunk size distribution
        chunk_sizes = [chunk.char_count for chunk in processed_doc.chunks]
        print(f"📊 Chunk size distribution:")
        print(f"   - Min: {min(chunk_sizes)} chars")
        print(f"   - Max: {max(chunk_sizes)} chars")
        print(f"   - Average: {sum(chunk_sizes) // len(chunk_sizes)} chars")
        print()

        # Show first 3 chunks
        print(f"First 3 chunks preview:")
        for chunk in processed_doc.chunks[:3]:
            print(f"\n  Chunk #{chunk.chunk_id} ({chunk.char_count} chars, pages {chunk.page_numbers}):")
            if chunk.token_count is not None:
                print(f"    Tokens: {chunk.token_count}")
            print(f"  {chunk.content[:100].replace(chr(10), ' ')}...")

        print()
        print("=" * 80)
        print("✅ Custom chunk demo completed!")
        print("=" * 80)

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


def create_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description='PDF Processing Demo - Test chunking functionality with optional tokenizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (auto-detect tokenizer)
  python chunk_pdf.py document.pdf

  # Specify tokenizer by model serving name
  python chunk_pdf.py document.pdf --model Meta-Llama-3.1-8B-Instruct-Q4_K_M

  # Custom chunk size
  python chunk_pdf.py document.pdf --chunk-size 500 --chunk-overlap 100

  # Custom chunks with specific tokenizer
  python chunk_pdf.py document.pdf --chunk-size 2000 --chunk-overlap 400 --model MyModel

  # Custom upload directory
  python chunk_pdf.py document.pdf --upload-dir my_documents

  # All options combined
  python chunk_pdf.py document.pdf --model MyModel --chunk-size 1000 --upload-dir custom_dir

Available Models:
"""
    )

    # Show available models from config
    try:
        if settings.models:
            epilog_lines = []
            for i, model in enumerate(settings.models, 1):
                name = model.serving_name or model.model_name_or_path
                tokenizer_info = f"✓ tokenizer: {model.tokenizer}" if model.tokenizer else "✗ no tokenizer"
                epilog_lines.append(f"  {i}. {name} ({tokenizer_info})")
            parser.epilog += "\n".join(epilog_lines)
    except:
        pass

    parser.add_argument(
        'pdf_path',
        type=str,
        help='Path to the PDF file to process'
    )

    parser.add_argument(
        '-m', '--model',
        type=str,
        default=None,
        dest='model_identifier',
        help='Model serving_name or model_name_or_path to use for tokenizer (default: auto-detect first model with tokenizer)'
    )

    parser.add_argument(
        '--chunk-size',
        type=int,
        default=None,
        help='Custom chunk size in characters (default: from config, usually 1000)'
    )

    parser.add_argument(
        '--chunk-overlap',
        type=int,
        default=None,
        help='Custom chunk overlap in characters (default: from config, usually 200)'
    )

    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List available models and their tokenizer status'
    )


    parser.add_argument(
        '--upload-dir',
        type=str,
        default='demo_uploads',
        help='Directory to store uploaded documents and metadata (default: demo_uploads)'
    )

    return parser


def list_models():
    """List available models and their tokenizer configuration"""
    print("=" * 80)
    print("Available Models")
    print("=" * 80)
    print()

    if not settings.models:
        print("⚠️  No models configured in env.yaml")
        return

    for i, model in enumerate(settings.models, 1):
        print(f"{i}. Model Configuration:")
        print(f"   model_name_or_path: {model.model_name_or_path}")
        print(f"   serving_name: {model.serving_name or '(not set)'}")
        if model.tokenizer:
            print(f"   tokenizer: ✓ {model.tokenizer}")
        else:
            print(f"   tokenizer: ✗ Not configured")
        print()

    # Show which tokenizer will be auto-detected
    auto_tokenizer = settings.get_tokenizer_for_model()
    if auto_tokenizer:
        print(f"🔧 Auto-detected tokenizer: {auto_tokenizer}")
    else:
        print(f"ℹ️  No tokenizer will be used (token_count will be None)")
    print()


def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()

    # Handle list models
    if args.list_models:
        list_models()
        return

    # Validate PDF path
    pdf_path = args.pdf_path

    # Check if using custom chunk settings
    if args.chunk_size is not None or args.chunk_overlap is not None:
        chunk_size = args.chunk_size or settings.documents.chunk_size
        chunk_overlap = args.chunk_overlap or settings.documents.chunk_overlap

        if chunk_size <= 0:
            print("❌ Error: Chunk size must be > 0")
            sys.exit(1)

        if chunk_overlap < 0:
            print("❌ Error: Chunk overlap must be >= 0")
            sys.exit(1)

        if chunk_overlap >= chunk_size:
            print("❌ Error: Chunk overlap must be < chunk size")
            sys.exit(1)

        asyncio.run(demo_with_custom_chunk_size(
            pdf_path,
            chunk_size,
            chunk_overlap,
            args.model_identifier,
            args.upload_dir
        ))
    else:
        asyncio.run(demo_pdf_processing(pdf_path, args.model_identifier, args.upload_dir))


if __name__ == "__main__":
    main()

