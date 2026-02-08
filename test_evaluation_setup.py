#!/usr/bin/env python3
"""
Test script to verify evaluation system installation and basic functionality
"""
import sys
import os

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    try:
        from src.evaluation import QAGenerator, RAGEvaluator, QAStorage
        print("✓ Evaluation modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_corpus_exists():
    """Test that corpus file exists."""
    print("\nChecking corpus file...")
    corpus_paths = [
        "./data/fixed_wiki_pages.json",
        "./data/random_wiki_pages.json"
    ]
    
    found = False
    for path in corpus_paths:
        if os.path.exists(path):
            print(f"✓ Found corpus: {path}")
            found = True
            break
    
    if not found:
        print("✗ No corpus file found. Please run indexing first.")
        return False
    
    return True

def test_storage_directory():
    """Test that storage directory can be created."""
    print("\nChecking storage directory...")
    storage_dir = "./data/evaluation"
    
    try:
        os.makedirs(storage_dir, exist_ok=True)
        print(f"✓ Storage directory ready: {storage_dir}")
        return True
    except Exception as e:
        print(f"✗ Error creating storage directory: {e}")
        return False

def test_basic_functionality():
    """Test basic Q&A storage functionality."""
    print("\nTesting basic functionality...")
    try:
        from src.evaluation import QAStorage
        
        storage = QAStorage()
        
        # Test validation
        test_qa = {
            'question_id': 'TEST001',
            'question': 'What is a test?',
            'answer': 'A test is a verification procedure.',
            'question_type': 'factual',
            'source_ids': [12345],
            'source_urls': ['https://en.wikipedia.org/wiki/Test']
        }
        
        is_valid, error = storage.validate_qa_pair(test_qa)
        
        if is_valid:
            print("✓ Q&A validation working")
            return True
        else:
            print(f"✗ Validation failed: {error}")
            return False
    
    except Exception as e:
        print(f"✗ Error testing functionality: {e}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("EVALUATION SYSTEM - INSTALLATION TEST")
    print("="*60 + "\n")
    
    tests = [
        ("Module Imports", test_imports),
        ("Corpus File", test_corpus_exists),
        ("Storage Directory", test_storage_directory),
        ("Basic Functionality", test_basic_functionality)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8s} {test_name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed! Evaluation system is ready.")
        print("\nNext steps:")
        print("1. Generate Q&A dataset:")
        print("   python run_evaluation.py generate --total-questions 20")
        print("\n2. Run evaluation:")
        print("   python run_evaluation.py evaluate --dataset-file <filename>")
        print("\n3. Or run full pipeline:")
        print("   python run_evaluation.py full --total-questions 20")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("- Make sure you're in the project root directory")
        print("- Install requirements: pip install -r requirements.txt")
        print("- Run indexing first to create corpus files")
        return 1

if __name__ == "__main__":
    sys.exit(main())
