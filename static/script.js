async function compareEmbeddings() {
  const chunks = [
    document.getElementById("chunk1").value.trim(),
    document.getElementById("chunk2").value.trim(),
    document.getElementById("chunk3").value.trim()
  ];

  const status = document.getElementById("status");
  const button = document.getElementById("compareBtn");

  if (chunks.some(c => !c)) {
    status.textContent = "Please enter all 3 chunks.";
    return;
  }

  button.disabled = true;
  status.textContent = "Generating embeddings and calculating metrics...";

  try {
    const response = await fetch("/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ chunks })
    });

    const data = await response.json();

    if (!response.ok || data.error) {
      throw new Error(data.error || "Request failed");
    }

    document.getElementById("results").classList.remove("hidden");

    document.getElementById("miniDim").textContent =
      `${data.models.MiniLM.dimensions} dimensions`;

    document.getElementById("bgeDim").textContent =
      `${data.models.BGE.dimensions} dimensions`;

    renderMatrix("miniCosine", data.models.MiniLM.metrics.cosine_similarity);
    renderMatrix("bgeCosine", data.models.BGE.metrics.cosine_similarity);

    renderMatrix("miniEuclidean", data.models.MiniLM.metrics.euclidean_distance);
    renderMatrix("bgeEuclidean", data.models.BGE.metrics.euclidean_distance);

    renderMatrix("miniDot", data.models.MiniLM.metrics.dot_product);
    renderMatrix("bgeDot", data.models.BGE.metrics.dot_product);

    status.textContent = "Comparison complete.";
  } catch (error) {
    status.textContent = `Error: ${error.message}`;
  } finally {
    button.disabled = false;
  }
}

function renderMatrix(elementId, matrix) {
  const labels = ["S1", "S2", "S3"];

  let html = "<table><tr><th></th>";
  labels.forEach(label => html += `<th>${label}</th>`);
  html += "</tr>";

  matrix.forEach((row, i) => {
    html += `<tr><th>${labels[i]}</th>`;
    row.forEach(value => {
      html += `<td>${Number(value).toFixed(4)}</td>`;
    });
    html += "</tr>";
  });

  html += "</table>";
  document.getElementById(elementId).innerHTML = html;
}
