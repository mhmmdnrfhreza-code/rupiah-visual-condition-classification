import os
from pathlib import Path
from uuid import uuid4

from flask import Flask, flash, redirect, render_template, request, url_for
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from src.config import ALLOWED_IMAGE_EXTENSIONS, ARTIFACTS_DIR, DATASET_DIR, MAX_UPLOAD_SIZE, UPLOAD_FOLDER
from src.model_page import build_model_page_context
from src.predict_model import predict_image


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_mapping(
        SECRET_KEY=os.environ.get("FLASK_SECRET_KEY", "rvcc-local-dev"),
        UPLOAD_FOLDER=str(UPLOAD_FOLDER),
        ARTIFACTS_DIR=str(ARTIFACTS_DIR),
        DATASET_DIR=str(DATASET_DIR),
        MAX_CONTENT_LENGTH=MAX_UPLOAD_SIZE,
    )
    if test_config:
        app.config.update(test_config)

    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)

    @app.errorhandler(RequestEntityTooLarge)
    def handle_large_file(_error):
        flash("Ukuran file terlalu besar. Maksimal 8 MB.", "danger")
        return redirect(url_for("index"))

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/predict")
    def predict():
        uploaded_file = request.files.get("file")
        if uploaded_file is None:
            flash("File gambar belum dipilih.", "danger")
            return redirect(url_for("index"))

        if uploaded_file.filename == "":
            flash("File gambar belum dipilih.", "danger")
            return redirect(url_for("index"))

        if not allowed_file(uploaded_file.filename):
            flash("Format file tidak didukung. Gunakan PNG, JPG, atau JPEG.", "danger")
            return redirect(url_for("index"))

        original_name = secure_filename(uploaded_file.filename)
        suffix = Path(original_name).suffix.lower()
        filename = f"{Path(original_name).stem[:40]}-{uuid4().hex[:12]}{suffix}"
        upload_path = Path(app.config["UPLOAD_FOLDER"]) / filename
        uploaded_file.save(upload_path)

        try:
            result = predict_image(upload_path, app.config["ARTIFACTS_DIR"])
        except FileNotFoundError:
            upload_path.unlink(missing_ok=True)
            flash("Model belum tersedia. Jalankan training model terlebih dahulu.", "danger")
            return redirect(url_for("index"))
        except ValueError:
            upload_path.unlink(missing_ok=True)
            flash("Gambar gagal dibaca. Gunakan file PNG, JPG, atau JPEG yang valid.", "danger")
            return redirect(url_for("index"))
        except Exception:
            upload_path.unlink(missing_ok=True)
            app.logger.exception("Unexpected prediction error")
            flash("Gambar gagal diproses. Periksa file gambar atau coba gambar lain.", "danger")
            return redirect(url_for("index"))

        result["image_url"] = url_for("static", filename=f"uploads/{filename}")
        return render_template("result.html", result=result)

    @app.get("/about")
    def about():
        return render_template("about.html")

    @app.get("/model")
    def model_page():
        context = build_model_page_context(app.config["DATASET_DIR"], app.config["ARTIFACTS_DIR"])
        return render_template("model.html", model=context)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", use_reloader=False)
