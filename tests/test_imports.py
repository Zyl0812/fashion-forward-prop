def test_package_imports():
    import smart_image_similarity  # noqa: F401
    from smart_image_similarity.webapp.app import create_app  # noqa: F401
