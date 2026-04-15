# evaluation/test_dataset.py
# Ground truth Q&A pairs generated from razorpay_refund_policy.pdf
# Used by ragas_evaluator.py to benchmark the RAG pipeline.
# Format: { question, ground_truth }
# - question    : realistic user support query
# - ground_truth: the correct answer drawn verbatim / closely from the source doc

TEST_DATASET = [
    # ── RETURNS ──────────────────────────────────────────────────────────────
    {
        "question": "How many days do I have to return a product?",
        "ground_truth": (
            "You have 30 days from the date of purchase to return a product. "
            "If 30 days have gone by since your purchase, a refund or exchange cannot be offered."
        ),
    },
    {
        "question": "What condition must the item be in to be eligible for a return?",
        "ground_truth": (
            "The item must be unused and in the same condition that you received it. "
            "It must also be in the original packaging."
        ),
    },
    {
        "question": "Can I return perishable goods like food or flowers?",
        "ground_truth": (
            "No. Perishable goods such as food, flowers, newspapers, or magazines "
            "cannot be returned."
        ),
    },
    {
        "question": "What types of goods are exempt from being returned?",
        "ground_truth": (
            "Items exempt from return include: perishable goods (food, flowers, newspapers, magazines), "
            "intimate or sanitary goods, hazardous materials, flammable liquids or gases, "
            "gift cards, downloadable software products, and some health and personal care items."
        ),
    },
    {
        "question": "Do I need a receipt to complete a return?",
        "ground_truth": (
            "Yes. A receipt or proof of purchase is required to complete your return."
        ),
    },
    {
        "question": "Should I send my return back to the manufacturer?",
        "ground_truth": (
            "No. You should not send your purchase back to the manufacturer."
        ),
    },
    # ── PARTIAL REFUNDS ───────────────────────────────────────────────────────
    {
        "question": "In what situations is only a partial refund granted?",
        "ground_truth": (
            "Partial refunds may be granted for: a book with obvious signs of use; "
            "a CD, DVD, VHS tape, software, video game, cassette tape, or vinyl record that has been opened; "
            "any item not in its original condition, damaged, or missing parts for reasons not due to our error; "
            "and any item returned more than 30 days after delivery."
        ),
    },
    {
        "question": "Can I get a refund on an opened video game or DVD?",
        "ground_truth": (
            "Only a partial refund may be granted for a CD, DVD, VHS tape, software, "
            "video game, cassette tape, or vinyl record that has been opened."
        ),
    },
    # ── REFUND PROCESS ────────────────────────────────────────────────────────
    {
        "question": "What happens after my return is received?",
        "ground_truth": (
            "Once your return is received and inspected, you will be sent an email notifying you "
            "that the returned item has been received. You will also be notified of the approval "
            "or rejection of your refund."
        ),
    },
    {
        "question": "How will my refund be issued if it is approved?",
        "ground_truth": (
            "If approved, the refund will be processed and a credit will automatically be applied "
            "to your credit card or original method of payment within a certain number of days."
        ),
    },
    # ── LATE OR MISSING REFUNDS ───────────────────────────────────────────────
    {
        "question": "I was approved for a refund but haven't received it yet. What should I do?",
        "ground_truth": (
            "First, check your bank account again. Then contact your credit card company, as it may "
            "take some time before the refund is officially posted. Next, contact your bank, as there "
            "is often some processing time. If you have done all of this and still have not received "
            "your refund, contact support at contact@ezlogistics.in."
        ),
    },
    {
        "question": "Who should I contact if my refund is still missing after checking with my bank?",
        "ground_truth": (
            "If you have checked your bank account, contacted your credit card company, and contacted "
            "your bank, and still have not received your refund, please contact support at "
            "contact@ezlogistics.in."
        ),
    },
    # ── SALE ITEMS ────────────────────────────────────────────────────────────
    {
        "question": "Can I get a refund on a sale item?",
        "ground_truth": (
            "No. Only regular priced items may be refunded. Sale items cannot be refunded."
        ),
    },
    # ── EXCHANGES ─────────────────────────────────────────────────────────────
    {
        "question": "Under what circumstances can I exchange an item?",
        "ground_truth": (
            "Items are only replaced if they are defective or damaged. "
            "If you need to exchange it for the same item, send an email to contact@ezlogistics.in."
        ),
    },
    {
        "question": "Where do I send an item I want to exchange?",
        "ground_truth": (
            "Send the item to: Conexial Supply Chain India PVT LTD, #615, 8th Main, 7th Cross, "
            "HBR 3rd Block First Stage, Bangalore 560043."
        ),
    },
    # ── GIFTS ─────────────────────────────────────────────────────────────────
    {
        "question": "What happens if I return a gift that was shipped directly to me?",
        "ground_truth": (
            "If the item was marked as a gift when purchased and shipped directly to you, "
            "you will receive a gift credit for the value of your return. Once the returned item "
            "is received, a gift certificate will be mailed to you."
        ),
    },
    {
        "question": "What if the gift was not marked as a gift at the time of purchase?",
        "ground_truth": (
            "If the item was not marked as a gift when purchased, or the gift giver had the order "
            "shipped to themselves first, the refund will be sent to the gift giver, "
            "who will find out about your return."
        ),
    },
    # ── SHIPPING ──────────────────────────────────────────────────────────────
    {
        "question": "Where should I mail my return product?",
        "ground_truth": (
            "Mail your return to: Conexial Supply Chain India PVT LTD, #615, 8th Main, 7th Cross, "
            "HBR 3rd Block First Stage, Bangalore 560043."
        ),
    },
    {
        "question": "Who pays for return shipping costs?",
        "ground_truth": (
            "You are responsible for paying your own shipping costs for returning your item. "
            "Shipping costs are non-refundable. If you receive a refund, the cost of return "
            "shipping will be deducted from your refund."
        ),
    },
    {
        "question": "Should I use a trackable shipping service when returning an item?",
        "ground_truth": (
            "If you are shipping an item over 1000 rupees, you should consider using a trackable "
            "shipping service or purchasing shipping insurance, as receipt of the returned item "
            "cannot be guaranteed."
        ),
    },
]

# TEST_DATASET = TEST_DATASET[:5]