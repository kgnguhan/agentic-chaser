"""
Canonical LOA workflow state names. All states are descriptive so it is clear
who we are waiting on and what the next step is.
"""

# Client-side: waiting on client
AWAITING_CLIENT_SIGNATURE = "Awaiting Client Signature"  # LOA prepared, waiting for client to sign and submit
DOCUMENT_AWAITING_VERIFICATION = "Document Awaiting Verification"  # Client submitted a document; verification pending
CLIENT_DOCUMENTS_REJECTED = "Client Documents Rejected"  # Verification failed; client must resubmit documents

# After client: we have signed LOA, ready for provider
SIGNED_LOA_READY_FOR_PROVIDER = "Signed LOA - Ready for Provider"  # Client signed; ready to submit to provider

# Provider-side: waiting on provider
SUBMITTED_TO_PROVIDER = "Submitted to Provider"  # We sent LOA to provider; awaiting their response
WITH_PROVIDER_PROCESSING = "With Provider - Processing"  # Provider has it and is processing
PROVIDER_RESPONSE_INCOMPLETE = "Provider Response Incomplete"  # Provider replied but information was incomplete

# Hand-back: we have what we need from provider
PROVIDER_INFO_RECEIVED_NOTIFY_CLIENT = "Provider Info Received - Notify Client"  # Next step: notify client

# Terminal
CASE_COMPLETE = "Case Complete"

# Lists for filtering and validation
CLIENT_CHASE_STATES = (
    AWAITING_CLIENT_SIGNATURE,
    DOCUMENT_AWAITING_VERIFICATION,
    CLIENT_DOCUMENTS_REJECTED,
    PROVIDER_INFO_RECEIVED_NOTIFY_CLIENT,
)
PROVIDER_CHASE_STATES = (
    SUBMITTED_TO_PROVIDER,
    WITH_PROVIDER_PROCESSING,
    PROVIDER_RESPONSE_INCOMPLETE,
)
MARK_PROVIDER_INFO_RECEIVED_ALLOWED = PROVIDER_CHASE_STATES
LINK_DOCUMENT_ALLOWED_STATES = (AWAITING_CLIENT_SIGNATURE, CLIENT_DOCUMENTS_REJECTED)
DOCUMENT_VERIFICATION_STATE = DOCUMENT_AWAITING_VERIFICATION
DEFAULT_STATE = AWAITING_CLIENT_SIGNATURE
