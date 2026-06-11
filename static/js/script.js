const fileInput = document.querySelector("#file");
const previewImage = document.querySelector("#imagePreview");
const previewPlaceholder = document.querySelector("#previewPlaceholder");

if (fileInput && previewImage && previewPlaceholder) {
  fileInput.addEventListener("change", () => {
    const file = fileInput.files?.[0];
    if (!file) {
      previewImage.hidden = true;
      previewImage.removeAttribute("src");
      previewPlaceholder.hidden = false;
      return;
    }

    previewImage.src = URL.createObjectURL(file);
    previewImage.hidden = false;
    previewPlaceholder.hidden = true;
  });
}
