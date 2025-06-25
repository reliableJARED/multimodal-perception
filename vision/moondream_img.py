#https://moondream.ai/c/docs/advanced/transformers
"""
https://moondream.ai/c/docs/capabilities
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch
import cv2

camera_index = 0
camera = cv2.VideoCapture(camera_index)

# Check device availability and set appropriate dtype
if torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float16  # MPS works well with float16
else:
    device = "cpu"
    dtype = torch.float32  # CPU needs float32

print(f"Using device: {device}")
# Load the model
model = AutoModelForCausalLM.from_pretrained(
"vikhyatk/moondream2",
revision="2025-01-09",
trust_remote_code=True, # Uncomment for GPU acceleration & pip install accelerate # device_map={"": "cuda"}  
device_map={"": device},
torch_dtype=dtype
)

def capture_frame():
    """Capture a single frame from camera"""
        
    ret, frame = camera.read()
    if not ret:
        return None
        
    # Convert BGR to RGB for PIL
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    return pil_image
        
# Load your image
#image = Image.open("/path/to/your/image/image.png")
#Capture Fresh Image
image = capture_frame()

# 1. Use Image Directly, Captioning Example
print("Short caption:\n")
print(model.caption(image, length="short")["caption"])

print("Detailed caption Streaming:\n")
for t in model.caption(image, length="normal", stream=True)["caption"]:
    print(t, end="", flush=True)

print("\n\nUse Encoded Image for Multiple-Analysis - MUCH FASTER for multi analysis, no difference if one single operation on the image\n")
#For multiple operations on the same image, encode it once to save processing time:
encoded_image = model.encode_image(image)

# 1. Image Captioning, 
"""
Parameter	|   Type	                |   Description
------------------------------------------------------------
image       |PIL.Image or encoded image |The image to process
length      |str	                    |Caption detail level: "short" or "normal"
stream	    |bool	                    |Whether to stream the response token by token

RESPONSE FORMAT:
{
  "request_id": "2025-03-25_caption_2025-03-25-21:00:39-715d03",
  "caption": "A detailed caption describing the image..."
}
"""
print("Short caption:")
print(model.caption(encoded_image, length="short")["caption"])

print("Detailed caption:")
for t in model.caption(encoded_image, length="normal", stream=True)["caption"]:
    print(t, end="", flush=True)


# 2. Visual Question Answering
"""
Parameter	|   Type	                |   Description
------------------------------------------------------------
image       |PIL.Image or encoded image |The image to process
question    |str	                    |The question to ask about the image
stream	    |bool	                    |Whether to stream the response token by token

RESPONSE FORMAT:
  {
  "request_id": "2025-03-25_query_2025-03-25-21:00:39-715d03",
  "answer": "Detailed text answer to your question..."
  }
"""
print("Asking questions about the image:")
print(model.query(encoded_image, "Do you see a dog in the picture?")["answer"])




# 3. Object Detection
"""
Parameter	|   Type	                |   Description
------------------------------------------------------------
image       |PIL.Image or encoded image |The image to process
object_name |str	                    |The type of object to detect

RESPONSE FORMAT:
{
  "request_id": "2025-03-25_detect_2025-03-25-21:00:39-715d03",
  "objects": [
    {
      "x_min": 0.2,   // left boundary of detection box (normalized 0-1)
      "y_min": 0.3,   // top boundary of detection box (normalized 0-1)
      "x_max": 0.6,   // right boundary of detection box (normalized 0-1)
      "y_max": 0.8    // bottom boundary of detection box (normalized 0-1)
    },
    // Additional objects...
  ]
}
"""
print("Detecting Faces:")
#'face', 'person', 'car' or other common objects for the string input
objects = model.detect(encoded_image, "face")["objects"]
print(objects)
print(f"Found {len(objects)} face(s)")

print("Detecting ANY (a green dinosaur toy with a black cape) object:")
#detailed description of the object we want to find
#describing the object as specifically as possible for best results.
objects2 = model.detect(encoded_image, "a green dinosaur toy with a black cape")["objects"]
print(objects2)


# 4. Visual Pointing
"""
Parameter	|   Type	                |   Description
------------------------------------------------------------
image       |PIL.Image or encoded image |The image to process
object_name |str	                    |The type of object to locate


RESPONSE FORMAT:
{
  "request_id": "2025-03-25_point_2025-03-25-21:00:39-715d03",
  "points": [
    {
      "x": 0.65,      // x coordinate (normalized 0-1)
      "y": 0.42       // y coordinate (normalized 0-1)
    },
    // Additional points...
  ]
}
"""
print("Locating person objects:")
points = model.point(encoded_image, "person")["points"]
print(f"Found {len(points)} person(s)")

print("Locating ANY (blue couch with yellow pillows) objects:")
points2 = model.point(encoded_image, "blue couch with yellow pillows")["points"]
print(points2)

""" 
PERFORMANCE TESTs (MacbookPro M4 48GB RAM)

Direct Image Load v. Encoded Image Use
============================================================
PERFORMANCE SUMMARY
============================================================
Operation            Direct (s)   Encoded (s)   Speedup   
------------------------------------------------------------
caption_short        3.0908       1.6668        1.85      x
vqa                  1.8352       0.3994        4.59      x
detection            1.6168       0.1811        8.93      x
pointing             1.5639       0.1309        11.95     x
------------------------------------------------------------
TOTAL                8.1067       2.3782        3.41      x

"""