"""Ordered capture script driving the golden harness.

Ordered, not a bag: later entries depend on memories written by earlier
ones. The supersession pairs (flight date, meeting day) only exercise the
revision chain if the original was written first.

Content is deliberately note-shaped rather than conversational — short,
contextless fragments — because that is what the notes app will feed the
engine, and the extraction prompt is tuned for `[USER]/[ASSISTANT]` turns.
Divergence here is a finding, not a failure.
"""

from __future__ import annotations

from uuid import UUID

# Fixed so prompts and goldens are stable across runs. The runner purges
# this user before each pass, so reuse never leaks state between runs.
GOLDEN_USER_ID = UUID("11111111-2222-3333-4444-555555555555")

# (label, message). Labels name the golden files, so keep them stable.
CAPTURES: list[tuple[str, str]] = [
    ("meeting_action_item", "follow up with Rahul about the unpaid March invoice"),
    ("stray_idea", "idea: what if the inbox sorted itself by how likely you are to need it today"),
    ("cafe_wifi", "Blue Tokai wifi password is bluetokai2024, network BT-Guest-5G"),
    ("grocery_list", "groceries for saturday: milk, eggs, coffee beans, olive oil, spinach, greek yoghurt"),
    ("phone_number", "plumber Suresh 98201 44532, comes on sundays only"),
    ("flight_original", "flying to Delhi on the 11th, evening flight"),
    ("book_rec", "Priya recommended Piranesi by Susanna Clarke"),
    ("draft_email", "draft to landlord: need the lease renewal signed before the 30th, ask about the parking spot"),
    ("meeting_original", "sprint review moved to Wednesday 4pm"),
    ("gym_expiry", "gym membership expires Aug 30"),
    ("flight_superseded", "no wait, the Delhi flight is on the 14th not the 11th"),
    ("recurring_thought", "keep thinking I should write more, maybe a weekly essay"),
    ("meeting_superseded", "sprint review is Thursday now, not Wednesday"),
    ("action_item_two", "send the revised deck to Anita before friday standup"),
    ("preference", "I really don't like taking calls before 10am"),
    ("recurring_thought_two", "again with the writing thing — should just start a weekly essay"),
    ("location_fact", "moved to the Bandra place last month"),
    ("retraction", "lol ignore what I said about the parking spot, building has none"),
    ("recall_query", "what did I say about the Delhi trip?"),
    ("recurring_thought_three", "third time this month: weekly essay, just do it"),
]

# Read-only surfaces exercised after the write script, so the golden set
# covers retrieval and the revision chain as well as the write pipeline.
SEARCH_QUERIES: list[tuple[str, str]] = [
    ("search_delhi", "Delhi trip flight"),
    ("search_groceries", "what do I need to buy"),
    ("search_writing", "writing essays"),
]
