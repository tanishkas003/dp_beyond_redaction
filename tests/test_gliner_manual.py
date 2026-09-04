from src.detection import GLiNERDetector


text = """
Hi, I'm Priya Sharma.

I work as a Senior Software Engineer at Microsoft
in Bangalore.

My employee ID is EMP-12345.

We are currently working on Project Falcon.
"""


print("Loading GLiNER...")

detector = GLiNERDetector()

print("\nDetecting entities...\n")

entities = detector.detect(text)

for entity in entities:
    print(entity)