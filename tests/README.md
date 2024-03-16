# Testing with Pytest
We use pytest, a standard testing library for Python. You can learn more about it here https://docs.pytest.org/en/8.0.x/
## Running All Tests
Navigate to the root directory of your Python package in your terminal and execute the following command:

```pytest```

This command will discover all test files (usually named with test_*.py or *_test.py) within Syllabus and execute any test it finds. Most general tests are in the `tests` directory while unit tests are in separate files next to the files they test.

## Running Specific Tests

You can use various pytest options to filter and control which tests are run:

- By File: Run tests in a specific file:

    ```pytest tests/my_test_file.py```

- By Test Name: Run a particular test function:

    ```pytest -k "test_specific_function"```

    (Replace test_specific_function with the actual test function name)
