const translateBtn = document.getElementById('translateBtn');
const pronounceBtn = document.getElementById('pronounceBtn');
const textInput = document.getElementById('textInput');
const languageSelect = document.getElementById('languageSelect');
const translationText = document.getElementById('translationText');
const sourceLabel = document.getElementById('sourceLabel');
const audioPlayer = document.getElementById('audioPlayer');
const outputDiv = document.getElementById('output');
const cloudContainer = document.getElementById('cloudContainer');

// Animate entrance
window.onload = () => {
  gsap.from("#mainCard", { duration: 1, y: 40, opacity: 0, ease: "power3.out" });
  loadWordCloud();
};

async function callTranslate(text, lang) {
  const res = await fetch('/api/translate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, language: lang })
  });
  return res.json();
}

async function callPronounce(text, tts_lang='en') {
  const res = await fetch('/api/pronounce', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, tts_lang })
  });
  return res.json();
}

translateBtn.addEventListener('click', async () => {
  const text = textInput.value.trim();
  const lang = languageSelect.value;
  if (!text) return;

  translationText.innerText = "Translating...";
  outputDiv.classList.add('show');
  outputDiv.classList.remove('hidden');

  const data = await callTranslate(text, lang);
  if (data.translation) {
    translationText.innerText = data.translation;
    sourceLabel.innerText = data.source;
  } else {
    translationText.innerText = data.message || "Not found.";
    sourceLabel.innerText = data.source;
  }

  gsap.from("#translationText", { duration: 0.8, y: 20, opacity: 0, ease: "power3.out" });
});

pronounceBtn.addEventListener('click', async () => {
  const text = textInput.value.trim();
  const lang = languageSelect.value;
  if (!text) return;

  const res = await callPronounce(text, lang);
  if (res.audio_url) {
    audioPlayer.src = res.audio_url;
    audioPlayer.classList.remove('hidden');
    audioPlayer.play();
    gsap.from("#audioPlayer", { duration: 0.8, scale: 0.8, opacity: 0, ease: "back.out(1.7)" });
  }
});

// Load word cloud
async function loadWordCloud() {
  try {
    const res = await fetch('/api/words');
    const data = await res.json();
    if (data.words) {
      cloudContainer.innerHTML = '';
      data.words.forEach(word => {
        const span = document.createElement('span');
        span.textContent = word;
        span.className = "wordCloudItem";
        cloudContainer.appendChild(span);

        // click → auto-fill and translate
        span.addEventListener('click', async () => {
          textInput.value = word;
          translateBtn.click();
        });
      });
    }
  } catch (err) {
    console.error("Word cloud load failed:", err);
  }
}
