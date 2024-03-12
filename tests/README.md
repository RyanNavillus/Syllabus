# Testing with Pytest
Pytest is very powerfull to test our codes.
## Running All Tests
Navigate to the root directory of your Python package in your terminal and execute the following command:

```pytest```

This command will discover all test files (usually named with test_*.py or *_test.py conventions) and subdirectories within your package and execute all the tests it finds.

## Running Specific Tests

You can use various pytest options to filter and control which tests are run:

- By File: Run tests in a specific file:

    ```pytest tests/my_test_file.py```

- By Test Name: Run a particular test function:

    ```pytest -k "test_specific_function"```

    (Replace test_specific_function with the actual test function name)
