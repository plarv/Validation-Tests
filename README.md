Validation Tests : Z
These tests are designed to run without the SDK for ease of use. To get started, obtain an API key from https://argus.plarv.com/dashboard/api-keys and replace the placeholder key in the test file.

Note: The official SDK includes a zero-block policy and async prefetch. This test suite omits both for simplicity.


NOTE: To ensure results reflect genuine model behavior rather than cherry-picked conditions, we run each test by selecting a seed from a pool of approximately 10,000 random seeds. Tests are two-stage: first establishing that a real anomaly exists, then confirming Argus detected and contained it. Because seeds are drawn from a large pool, both stages don't always align in a single run. If a stage-two condition isn't met, re-run the test. If Argus itself appears to make an error, contact us at contact@plarv.com.

Step Time Considerations
Argus is optimized for training runs with step times above 300ms. Models with fast step times — such as GPT-2 Small — will complete runs quickly under these tests.

Argus is currently activated in the Americas by default. If you are in another region and would like Argus enabled for your location, contact us at contact@plarv.com with your estimated step time and region.

Early Detection Behavior
In some tests, Argus may flag a collapse before it fully materializes. This is not an error — it reflects Argus's ability to detect collapse trajectories in advance through its internal detection engine. If a test reports an unexpected alert, inspect the training signal rather than dismissing it.
