import torch
import torch.nn.functional as F


class BaseTrainer:
    def __init__(self, model, optimizer, train_dataloader, device, epochs=1, val_dataloader=None,
                 max_length=512):
        self.tokenizer = model.tokenizer
        self.classifier = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.epochs = epochs
        self.max_length = max_length

    def train(self):
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in self.train_dataloader:
                self.optimizer.zero_grad()
                texts, labels = batch
                labels = labels.to(self.device)
                inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt',
                                        max_length=self.max_length)
                input_ids = inputs["input_ids"].to(self.device)
                if self.classifier.uses_attention:
                    attention_mask = inputs["attention_mask"].to(self.device)
                    outputs = self.classifier(input_ids, attention_mask=attention_mask)
                else:
                    outputs = self.classifier(input_ids)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            avg_train_loss = total_loss / len(self.train_dataloader)
            val_acc = "None"
            if self.val_dataloader is not None:
                val_acc = self.evaluate(self.val_dataloader)
            print(f'Epoch {epoch + 1}/{self.epochs}, Train Loss: {avg_train_loss}, Validation Accuracy: {val_acc}')

    def evaluate(self, dataloader):
        total_correct = 0
        total_examples = 0
        with torch.no_grad():
            for batch in dataloader:
                texts, labels = batch
                inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt',
                                        max_length=self.max_length)
                input_ids = inputs["input_ids"].to(self.device)
                labels = labels.to(self.device)
                if self.classifier.uses_attention:
                    attention_mask = inputs["attention_mask"].to(self.device)
                    outputs = self.classifier(input_ids, attention_mask=attention_mask)
                else:
                    outputs = self.classifier(input_ids)
                predictions = torch.argmax(outputs, dim=-1)
                correct = (predictions == labels).sum().item()
                total_correct += correct
                total_examples += labels.size(0)
        return total_correct / total_examples


class SupervisedTrainer(BaseTrainer):
    def __init__(self, model, criterion, optimizer, train_dataloader, device="gpu", epochs=1, val_dataloader=None,
                 max_length=512):
        super().__init__(model, optimizer, train_dataloader, device, epochs, val_dataloader, max_length)
        self.criterion = criterion


class UDATrainer(BaseTrainer):
    def __init__(self, model, augmenter, optimizer, supervised_dataloader, unsupervised_dataloader, device="gpu",
                 epochs=1, val_dataloader=None, max_length=512):
        super().__init__(model, optimizer, supervised_dataloader, device, epochs, val_dataloader, max_length)
        self.unsupervised_dataloader = unsupervised_dataloader
        self.uda_coeff = 1.0
        self.criterion = torch.nn.CrossEntropyLoss()
        self.augmenter = augmenter

    def train(self):
        for epoch in range(self.epochs):
            total_loss = 0
            for batch, unsup_batch in zip(self.train_dataloader, self.unsupervised_dataloader):
                texts, labels = batch
                labels = labels.to(self.device)

                # Tokenize labeled data
                encoded_inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt',
                                                max_length=self.max_length)
                input_ids = encoded_inputs['input_ids'].to(self.device)
                if self.classifier.uses_attention:
                    attention_mask = encoded_inputs['attention_mask'].to(self.device)

                # Unsupervised data
                unsup_texts, _ = unsup_batch

                unsup_augmented_texts = self.augmenter.augment(unsup_texts)

                # Tokenize unsupervised data
                unsup_encoded_inputs = self.tokenizer(unsup_texts, padding=True, truncation=True, return_tensors='pt',
                                                      max_length=self.max_length)
                unsup_input_ids = unsup_encoded_inputs['input_ids'].to(self.device)
                if self.classifier.uses_attention:
                    unsup_attention_mask = unsup_encoded_inputs['attention_mask'].to(self.device)
                unsup_aug_encoded_inputs = self.tokenizer(unsup_augmented_texts, padding=True, truncation=True,
                                                          return_tensors='pt', max_length=self.max_length)

                # Tokenize augmented unsupervised data
                unsup_aug_input_ids = unsup_aug_encoded_inputs['input_ids'].to(self.device)
                if self.classifier.uses_attention:
                    unsup_aug_attention_mask = unsup_aug_encoded_inputs['attention_mask'].to(self.device)

                self.optimizer.zero_grad()

                # Labeled data forward pass
                if self.classifier.uses_attention:
                    outputs = self.classifier(input_ids, attention_mask=attention_mask)
                else:
                    outputs = self.classifier(input_ids)
                supervised_loss = self.criterion(outputs, labels)

                # Unsupervised data forward pass
                if self.classifier.uses_attention:
                    unsup_outputs = self.classifier(unsup_input_ids, attention_mask=unsup_attention_mask)
                else:
                    unsup_outputs = self.classifier(unsup_input_ids)
                unsup_probs = F.softmax(unsup_outputs, dim=1)

                # Augmented unsupervised data forward pass
                if self.classifier.uses_attention:
                    unsup_aug_outputs = self.classifier(unsup_aug_input_ids, attention_mask=unsup_aug_attention_mask)
                else:
                    unsup_aug_outputs = self.classifier(unsup_aug_input_ids)
                unsup_aug_probs = F.softmax(unsup_aug_outputs, dim=1)

                # Consistency loss
                consistency_loss = F.kl_div(unsup_probs.log_softmax(dim=1), unsup_aug_probs.softmax(dim=1),
                                            reduction='batchmean')

                # Combined loss and optimizer step
                combined_loss = supervised_loss + self.uda_coeff * consistency_loss
                total_loss += combined_loss.item()
                combined_loss.backward()
                self.optimizer.step()

            avg_train_loss = total_loss / len(self.train_dataloader)
            val_acc = "None"
            if self.val_dataloader is not None:
                val_acc = self.evaluate(self.val_dataloader)
            print(f'Epoch {epoch + 1}/{self.epochs}, Train Loss: {avg_train_loss}, Validation Accuracy: {val_acc}')
