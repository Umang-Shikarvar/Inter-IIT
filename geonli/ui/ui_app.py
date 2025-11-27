import streamlit as st
import requests, tempfile

st.title("🌍 GeoNLI – Satellite AI")
url = "http://localhost:8000/geonli/task"

file = st.file_uploader("Upload Satellite Image", type=["png","jpg","jpeg"])

if file:
    st.image(file)

    task = st.radio(
        "Select Task:",
        ["Caption", "VQA", "Bounding Box (Grounding)"]
    )

    # Inputs based on task
    question = ""
    grounding = ""

    if task == "VQA":
        question = st.text_input("Ask a VQA Question:")
    elif task == "Bounding Box (Grounding)":
        grounding = st.text_input("Object to detect (e.g. 'car', 'building')")

    if st.button("🚀 Run Task"):
        with tempfile.NamedTemporaryFile(delete=False) as t:
            t.write(file.read())
            t.flush()

        task_map = {
            "Caption": "caption",
            "VQA": "vqa",
            "Bounding Box (Grounding)": "ground"
        }
        data = {"task": task_map[task]}
        data["question"] = question
        data["grounding"] = grounding

        files = {"image": (file.name, open(t.name, "rb"), file.type)}
        r = requests.post(url, files=files, data=data)

        try:
            if task == "Bounding Box (Grounding)":
                st.image(r.content, caption="Detected Objects")
            else:
                st.json(r.json())
        except:
            st.error("❗ Server returned invalid JSON")
            st.write(r.text)