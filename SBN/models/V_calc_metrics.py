class Metrics_V:
    def __init__(self):
        pass

    def _drop_duplicates(self, samples):
        return {sample_id: set(aspects) for sample_id, aspects in samples.items()}

    def calculate(self, true, predicted):
        true = self._drop_duplicates(true)
        predicted = self._drop_duplicates(predicted)

        assert len(true) == len(predicted)

        self.tp = 0
        self.fp = 0
        self.fn = 0
        for sample_id in (set(true) | set(predicted)):
            if sample_id not in true:
                raise ValueError(f"Sample id {sample_id} is not found in true aspects")
            if sample_id not in predicted:
                raise ValueError(f"Sample id {sample_id} is not found in predicted aspects")

            true_aspects = true[sample_id]
            pred_aspects = predicted[sample_id]

            current_tp = sum(pred_aspect in true_aspects for pred_aspect in pred_aspects)
            current_fp = len(pred_aspects) - current_tp
            current_fn = sum(true_aspect not in pred_aspects for true_aspect in true_aspects)
            
            self.tp += current_tp
            self.fp += current_fp
            self.fn += current_fn

        # return self.precision()

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp)
    
    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn)

    @property
    def f1(self) -> float:
        return 2 * self.precision * self.recall / (self.precision + self.recall)