# Synthetic incident: payment timeout

The payment adapter returned a timeout after a synthetic authorization request. The service did not receive a final response, so the payment state remained pending. The retry policy uses an idempotency key and checks transaction status before submitting another authorization. No customer or bank data is present in this example.
