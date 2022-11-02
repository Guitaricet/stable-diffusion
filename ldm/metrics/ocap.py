import torch
import evaluate
import easyocr
from tqdm.auto import tqdm


class CraftOCap:
    parameters = {
        "name": "craft-ocap",
        "beta": 0,
        "char_order": 4,
        "word_order": 0,
        "confidence_threshold": 0.5,
    }

    def __init__(self, use_gpu=None):
        if use_gpu is None:
            use_gpu = torch.cuda.is_available()

        self.ocr = easyocr.Reader(["en"], detect_network='craft', gpu=use_gpu)
        self.chrf = evaluate.load("chrf")
    
    @property
    def signature(self):
        n = self.parameters["name"]
        b = self.parameters["beta"]
        c = self.parameters["char_order"]
        w = self.parameters["word_order"]
        t = self.parameters["confidence_threshold"]
        return f"{n}-b{b}-c{c}-w{w}-t{t}"

    def compute(self, *, images=None, captions=None, progress_bar=True):
        """
        Args:
            images: a numpy array of shape (N, H, W, 3)
            captions: list of captions
        """
        assert len(images) == len(captions)
        ocr_features = self._extract_ocr_features(images, progress_bar=progress_bar)
        return self._compute_score_given_ocr_features(ocr_features, captions)

    def _compute_score_given_ocr_features(self, ocr_features, captions):
        ocr_texts = []
        for ocr_output in ocr_features:
            full_ocr_text = ""
            for ocr_item in ocr_output:
                ocr_confidence = ocr_item[2]
                if ocr_confidence < self.parameters["confidence_threshold"]:
                    continue

                ocr_text = ocr_item[1].lower()
                full_ocr_text += ocr_text + " "

            ocr_texts.append(full_ocr_text)

        # beta=0, char_order=4, word_order=0
        # TODO: label images manually and optimizer these haprams and confidence threshold
        # on a scale form 0 to 10, how well the in-schene text on the image follows the caption
        # (e.g., if a text2image model gave you a picture of "fireworks saying happy birthday" and the fireworks do say "happy birthday" the score would be 10)
        # (or, score is 0 if the text from the caption does not appear in the image at all)
        score = self.chrf.compute(
            predictions=ocr_texts,
            references=captions,
            beta=self.parameters["beta"],
            char_order=self.parameters["char_order"],
            word_order=self.parameters["word_order"],
        )
        return score

    def _extract_ocr_features(self, images, progress_bar=True):
        ocr_features = []
        for image in tqdm(images, disable=not progress_bar):
            ocr_output = self.ocr.readtext(
                image=image,
                batch_size=64,
            )
            ocr_features.append(ocr_output)
        return ocr_features
