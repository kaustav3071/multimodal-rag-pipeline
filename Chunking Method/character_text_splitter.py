from langchain_text_splitters.character import CharacterTextSplitter, RecursiveCharacterTextSplitter

text="""
The Model Y became the best-selling vehicle globally, with 350,000 units sold.

Production Challenges

Supply chain issues caused a 12% increase in production costs.

This is one very long paragraph that definitely exceeds our 100 character limit and has no double newlines inside it whatsoever, written continuously to demonstrate how a large block of text typically looks when extracted from a source and verified for length compliance.
"""


# splitter1 = CharacterTextSplitter(
#     chunk_size=100,
#     chunk_overlap=0,
#     separator=" " #["\n\n", "\n", ".", " ","\n\n\n"]
# )

# chunks1 = splitter1.split_text(text)
# for i, chunk in enumerate(chunks1):
#     print(f"Chunk {i+1}: ({len(chunk)} characters)")
#     print(chunk)


print("\n"+"="*60)
print("2. Recursive Character Text Splitter Solution")
print("="*60)

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separators=["\n\n", "\n", ".", " ", "","\n\n\n"]
)

chunks2 = recursive_splitter.split_text(text)
for i, chunk in enumerate(chunks2,1):
    print(f"Chunk {i}: ({len(chunk)} characters)")
    print(chunk)
    print()


