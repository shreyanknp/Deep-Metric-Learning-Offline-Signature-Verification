def test_imports():
    import src
    from src.utils import load_config  # type: ignore

    assert hasattr(src, "__version__")
