
import streamlit as st
import requests
import base64

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Pothole Detection System",
    page_icon="🚧",
    layout="wide"
)

st.title("🚧 AI Pothole Detection System")

st.write(
    "Upload a road image or video to detect potholes using YOLO."
)

st.divider()

# Image and Video Upload
uploaded_file = st.file_uploader(
    "Upload Image or Video",
    type=[
        "jpg", "jpeg", "png",
        "mp4", "avi", "mov", "mkv"
    ]
)

if uploaded_file is not None:

    file_name = uploaded_file.name.lower()

    is_video = file_name.endswith(
        (".mp4", ".avi", ".mov", ".mkv")
    )

    # Preview uploaded file
    if is_video:

        st.subheader("Uploaded Video")
        st.video(uploaded_file)

        endpoint = "/predict-video"

    else:

        st.subheader("Uploaded Image")
        st.image(
            uploaded_file,
            use_container_width=True
        )

        endpoint = "/predict"

    st.divider()

    if st.button(
        "🔍 Detect Potholes",
        type="primary",
        use_container_width=True
    ):

        with st.spinner("YOLO is detecting potholes..."):

            try:

                response = requests.post(
                    API_URL + endpoint,
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type
                        )
                    },
                    timeout=600
                )

                response.raise_for_status()

                # IMAGE RESULT
                if not is_video:

                    result = response.json()

                    if "error" in result:
                        st.error(result["error"])

                    else:

                        st.success("Image detection completed!")

                        st.metric(
                            "Total Potholes Detected",
                            result["total_potholes"]
                        )

                        image_data = base64.b64decode(
                            result["image"]
                        )

                        st.subheader("Detection Result")

                        st.image(
                            image_data,
                            use_container_width=True
                        )

                        if result["detections"]:

                            st.subheader("Detection Details")

                            st.dataframe(
                                result["detections"],
                                use_container_width=True
                            )

                        else:
                            st.info("No potholes detected.")

                # VIDEO RESULT
                else:

                    st.success("Video detection completed!")

                    total_frames = response.headers.get(
                        "X-Total-Frames", "N/A"
                    )

                    total_detections = response.headers.get(
                        "X-Total-Detections", "N/A"
                    )

                    col1, col2 = st.columns(2)

                    col1.metric(
                        "Frames Processed",
                        total_frames
                    )

                    col2.metric(
                        "Total Detections",
                        total_detections
                    )

                    st.subheader("Processed Video")

                    st.video(
                        response.content,
                        format="video/mp4"
                    )

                    st.download_button(
                        label="⬇️ Download Processed Video",
                        data=response.content,
                        file_name="pothole_detection.mp4",
                        mime="video/mp4"
                    )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Backend is not running. "
                    "Please start FastAPI first."
                )

            except requests.exceptions.Timeout:

                st.error(
                    "Processing timeout. Try a shorter video."
                )

            except requests.exceptions.HTTPError as e:

                st.error(
                    f"Backend Error: {e.response.text}"
                )

            except Exception as e:

                st.error(f"Something went wrong: {str(e)}")
