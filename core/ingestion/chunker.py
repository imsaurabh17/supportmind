import hashlib
import logging
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

def chunk_documents(
        documents: List[Document],
        chunk_size: int = 1000,
        chunk_overlap: int = 200
) -> List[Document]:
    """Split documents into overlapping chunks with unique IDs.

    Args:
        documents: List of Document objects to split.
        chunk_size: Maximum character length of each chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        List of chunked Document objects with enriched metadata.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    # Assign unique chunk_id to each chunk
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        raw = f"{source}-{i}-{chunk.page_content[:50]}"
        chunk.metadata["chunk_id"] = hashlib.md5(raw.encode()).hexdigest()[:12]

        chunk.metadata["chunk_index"] = i

    logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks

if __name__=="__main__":
    print(chunk_documents(documents=[Document(metadata={'producer': 'Microsoft® Word for Microsoft 365', 'creator': 'Microsoft® Word for Microsoft 365', 'creationdate': '2022-07-27T09:36:50+00:00', 'moddate': '2022-07-27T09:36:50+00:00', 'source': 'razorpay_refund_policy.pdf', 'total_pages': 3, 'page': 0, 'page_label': '1', 'file_path': 'D:\\supportmind\\supportmind\\data\\raw\\razorpay_refund_policy.pdf', 'doc_type': 'pdf'}, page_content='Refund & Cancellation \nPolicy for Payment Through \nRazorpay \n \n \n \nReturns \nOur policy lasts 30 days. If 30 days have gone by since your purchase, unfortunately we can’t offer you a \nrefund or exchange. \nTo be eligible for a return, your item must be unused and in the same condition that you received it. It must \nalso be in the original packaging. \nSeveral types of goods are exempt from being returned. Perishable goods such as food, flowers, newspapers \nor magazines cannot be returned. We also do not accept products  that are intimate or sanitary goods, \nhazardous materials, or flammable liquids or gases. \nAdditional non-returnable items: \n• Gift cards \n• Downloadable software products \n• Some health and personal care items \nTo complete your return, we require a receipt or proof of purchase. \nPlease do not send your purchase back to the manufacturer. \nThere are certain situations where only partial refunds are granted: (if applicable)  \nBook with obvious signs of use'), Document(metadata={'producer': 'Microsoft® Word for Microsoft 365', 'creator': 'Microsoft® Word for Microsoft 365', 'creationdate': '2022-07-27T09:36:50+00:00', 'moddate': '2022-07-27T09:36:50+00:00', 'source': 'razorpay_refund_policy.pdf', 'total_pages': 3, 'page': 1, 'page_label': '2', 'file_path': 'D:\\supportmind\\supportmind\\data\\raw\\razorpay_refund_policy.pdf', 'doc_type': 'pdf'}, page_content='CD, DVD, VHS tape, software, video game, cassette tape, or vinyl record that has been opened. \nAny item not in its original condition, is damaged or missing parts for reasons not due to our error.  \nAny item that is returned more than 30 days after delivery \nRefunds (if applicable) \nOnce your return is received and inspected, w e will send you an email to notify you that we have received \nyour returned item. We will also notify you of the approval or rejection of your refund.  \nIf you are approved, then your refund will be processed, and a credit will automatically be applied to your \ncredit card or original method of payment, within a certain amount of days. \nLate or missing refunds (if applicable) \nIf you haven’t received a refund yet, first check your bank account again. \nThen contact your credit card company, it may take some time before your refund is officially posted. \nNext contact your bank. There is often some processing time before a refund is posted.  \nIf you’ve done all of this and you still have not received your refund yet, please contact us at \ncontact@ezlogistics.in \nSale items (if applicable) \nOnly regular priced items may be refunded, unfortunately sale items cannot be refunded.  \nExchanges (if applicable) \nWe only replace items if they are defective or damaged. If you need to exchange it for the same item, send \nus an email at contact@ezlogistics.in and send your item to. Conexial Supply Chain India PVT LTD ,# 615,8TH \nMAIN ,7TH CROSS ,HBR 3RD BLOCK FIRST STAGE,BANGALORE 560043 \nGifts'), Document(metadata={'producer': 'Microsoft® Word for Microsoft 365', 'creator': 'Microsoft® Word for Microsoft 365', 'creationdate': '2022-07-27T09:36:50+00:00', 'moddate': '2022-07-27T09:36:50+00:00', 'source': 'razorpay_refund_policy.pdf', 'total_pages': 3, 'page': 2, 'page_label': '3', 'file_path': 'D:\\supportmind\\supportmind\\data\\raw\\razorpay_refund_policy.pdf', 'doc_type': 'pdf'}, page_content='If the item was marked as a gift when purchased and shipped directly to you, you’ll receive a gif t credit for \nthe value of your return. Once the returned item is received, a gift certificate will be mailed to you.  \nIf the item wasn’t marked as a gift when purchased, or the gift giver had the order shipped to themselves \nto give to you later, we will send a refund to the gift giver and he will find out about your return.  \nShipping \nTo return your product, you should mail your product to: Conexial Supply Chain India PVT LTD ,# 615,8TH \nMAIN ,7TH CROSS ,HBR 3RD BLOCK FIRST STAGE,BANGALORE 560043 \nYou will be responsible for paying for your own shipping costs for returning your item. Shipping costs are \nnon-refundable. If you receive a refund, the cost of return shipping will be deducted from your refund.  \nDepending on where you live, the time it may take for your exchanged product to reach you, may vary. \nIf you are shipping an item over 1000 rupees, you should consider using a trackable shipping service or \npurchasing shipping insurance. We don’t guarantee that we will receive your returned item.')]))