"""
tools.py - Mock Lead Capture Tool

Simulates capturing a lead by printing the collected information.
In production, this would integrate with a CRM or database.
"""


def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """
    Simulate capturing a lead with the collected user details.

    Args:
        name: The lead's full name.
        email: The lead's email address.
        platform: The lead's preferred social media or content platform.

    Returns:
        A confirmation string indicating the lead was captured.
    """
    print(f"\n{'='*50}")
    print(f"  ✅ Lead captured successfully!")
    print(f"  Name:     {name}")
    print(f"  Email:    {email}")
    print(f"  Platform: {platform}")
    print(f"{'='*50}\n")
    return f"Lead successfully captured for {name}"
