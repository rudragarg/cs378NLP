# This list of imports is likely incomplete --- add anything you need.
# TODO: Your code here.
import torch.nn as nn
import allennlp
from allennlp.nn.util import replace_masked_values, masked_log_softmax
import torch

class RNN(nn.Module):
    def __init__(self, embedding_matrix, hidden_size,
                 dropout):
        """
        Parameters
        ----------
        embedding_matrix: FloatTensor
            FloatTensor matrix of shape (num_words, embedding_dim),
            where each row of the matrix is a word vector for the
            associated word index.

        hidden_size: int
            The size of the hidden state in the RNN.

        dropout: float
            The dropout rate.
        """
        # Save the construction arguments, useful for serialization
        self.init_arguments = locals()
        self.init_arguments.pop("self")
        self.init_arguments.pop("__class__")
        super(RNN, self).__init__()

        #using CBOW as context
        self.embedding_matrix = embedding_matrix
        self.num_embedding_words = embedding_matrix.size(0)
        self.embedding_dim = embedding_matrix.size(1)


        # Create Embedding object
        self.embedding = nn.Embedding(self.num_embedding_words,
                                      self.embedding_dim, padding_idx=0)

        # Load our embedding matrix weights into the Embedding object,
        # and make them untrainable (requires_grad=False)
        self.embedding.weight = nn.Parameter(self.embedding_matrix,
                                             requires_grad=False)

        # Make a GRU to encode the passage. Note that batch_first=True.
        self.gruPassage = torch.nn.GRU(self.embedding_dim, hidden_size, num_layers=2, batch_first=True, bidirectional=True)

        # Make a GRU to encode the question. Note that batch_first=True.
        self.gruQuestion = torch.nn.GRU(self.embedding_dim, hidden_size, num_layers=2, batch_first=True, bidirectional=True)

        #Affine transform for predicting start index.
        self.start_output_projection = nn.Linear(6 * hidden_size, 1)

        # Affine transform for predicting end index.
        self.end_output_projection = nn.Linear(6 * hidden_size, 1)

        # Dropout layer
        self.dropout = torch.nn.Dropout(dropout)

        # Stores the number of gradient updates performed.
        self.global_step = 0

    def forward(self, passage, question):
        """
        The forward pass of the RNN-based model.

        Parameters
        ----------
        passage: Variable(LongTensor)
            A Variable(LongTensor) of shape (batch_size, passage_length)
            representing the words in the passage for each batch.

        question: Variable(LongTensor)
            A Variable(LongTensor) of shape (batch_size, question_length)
            representing the words in the question for each batch.

        Returns
        -------
        An output dictionary consisting of:
        start_logits: Variable(FloatTensor)
            The first element in the returned tuple. Variable(FloatTensor) of
            shape (batch_size, max_passage_size). Each value is the score
            assigned to a given token. Masked indices are assigned very
            small scores (-1e7).

        end_logits: Variable(FloatTensor)
            The second element in the returned tuple. Variable(FloatTensor) of
            shape (batch_size, max_passage_size). Each value is the score
            assigned to a given token. Masked indices are assigned very
            small scores (-1e7).

        softmax_start_logits: Variable(FloatTensor)
            The third element in the returned tuple. Variable(FloatTensor) of
            shape (batch_size, max_passage_size). Exactly the same as
            start_logits, but with a masked log softmax applied. Represents
            a probability distribution over the passage, indicating the
            probability that any given token is where the answer begins.
            Masked indices have probability mass of -inf.

        softmax_end_logits: Variable(FloatTensor)
            The fourth element in the returned tuple. Variable(FloatTensor) of
            shape (batch_size, max_passage_size). Exactly the same as
            start_logits, but with a masked log softmax applied. Represents
            a probability distribution over the passage, indicating the
            probability that any given token is where the answer end.
            Masked indices have probability mass of -inf.
        """
        # Mask: FloatTensor with 0 in positions that are
        # padding (word index 0) and 1 in positions with actual words.
        # Make a mask for the passage.  
        passage_mask = (passage != 0).type(
            torch.cuda.FloatTensor if passage.is_cuda else
            torch.FloatTensor)
        
        # Make a mask for the question.  
        question_mask = (question != 0).type(
            torch.cuda.FloatTensor if question.is_cuda else
            torch.FloatTensor)

        # Make a LongTensor with the length (number non-padding words
        # in) each passage.
    
        passage_lengths = (passage_mask.sum(dim=1)).type(
            torch.cuda.LongTensor if passage.is_cuda else
            torch.LongTensor)

        # Make a LongTensor with the length (number non-padding words
        # in) each question.
    
        #question_lengths = (question_mask.sum(dim=1)).type(torch.cuda.LongTensor)
        question_lengths = (question_mask.sum(dim=1)).type(
            torch.cuda.LongTensor if question.is_cuda else
            torch.LongTensor)

        # Part 1: Embed the passages and the questions.
        # 1.1. Embed the passage.
    
        embedded_passage = self.embedding(passage)

        # 1.2. Embed the question.
    
        embedded_question = self.embedding(question)

        # Part 2. Encode the embedded passages with the RNN.
        
        # 2.1. Sort embedded passages by decreasing order of passage_lengths.
        # Hint: allennlp.nn.util.sort_batch_by_length might be helpful.
        
        sorted_embedded_passage, sorted_sequence_lengths_passage, restoration_indices_passage, permutation_index_passage = allennlp.nn.util.sort_batch_by_length(tensor=embedded_passage, sequence_lengths=passage_lengths)
        sorted_sequence_lengths_passage = sorted_sequence_lengths_passage.type(torch.LongTensor)
        
        # 2.2. Pack the passages with torch.nn.utils.rnn.pack_padded_sequence.
        # Hint: Make sure you have the proper value for batch_first.
        
        padded_sequence_passage = torch.nn.utils.rnn.pack_padded_sequence(sorted_embedded_passage, sorted_sequence_lengths_passage, batch_first=True)
        
        # 2.3. Encode the packed passages with the RNN.
        
        encoded_passage, hidden_states_passage = self.gruPassage(padded_sequence_passage)

        # 2.4. Unpack (pad) the passages with
        # torch.nn.utils.rnn.pad_packed_sequence.
        # Hint: Make sure you have the proper value for batch_first.
    
        unpacked_passage, length_sequence_passage = torch.nn.utils.rnn.pad_packed_sequence(encoded_passage, batch_first=True)

        # 2.5. Unsort the unpacked, encoded passage to restore the
        # initial ordering.
        # Hint: Look into torch.index_select or NumPy/PyTorch fancy indexing.
    
        
        unsorted_passage = torch.index_select(unpacked_passage, 0, restoration_indices_passage, out=None)
        #unsorted_passage = unpacked_passage.index_select(0, restoration_indices_passage)

        # Part 3. Encode the embedded questions with the RNN.
        # 3.1. Sort the embedded questions by decreasing order
        # of question_lengths.
        # Hint: allennlp.nn.util.sort_batch_by_length might be helpful.
        sorted_embedded_question, sorted_sequence_lengths_question, restoration_indices_question, permutation_index_question = allennlp.nn.util.sort_batch_by_length(tensor=embedded_question, sequence_lengths=question_lengths)
      

        # 3.2. Pack the questions with pack_padded_sequence.
        # Hint: Make sure you have the proper value for batch_first.
        padded_sequence_question = torch.nn.utils.rnn.pack_padded_sequence(sorted_embedded_question, sorted_sequence_lengths_question, batch_first=True)

        # 3.3. Encode the questions with the RNN.
        encoded_question, hidden_states_question = self.gruQuestion(padded_sequence_question)

        # 3.4. Unpack (pad) the questions with pad_packed_sequence.
        # Hint: Make sure you have the proper value for batch_first.

        unpacked_question, length_sequence_question = torch.nn.utils.rnn.pad_packed_sequence(encoded_question, batch_first=True)

        # 3.5. Unsort the unpacked, encoded question to restore the
        # initial ordering.
        # Hint: Look into torch.index_select or NumPy/PyTorch fancy indexing.
    
        unsorted_question = torch.index_select(unpacked_question, 0, restoration_indices_question, out=None)
        #unsorted_question = unpacked_question.index_select(0, restoration_indices_question)

        # 3.6. Take the average of the GRU hidden states.
        # Hint: Be careful how you treat padding.
    
        average_question = (torch.sum(unsorted_question, dim=1) /
                     question_lengths.unsqueeze(1))

        # Part 4: Combine the passage and question representations by
        # concatenating the passage and question representations with
        # their product.

        # 4.1. Reshape the question encoding to make it
        # amenable to concatenation
    
        tiled_encoded_question = average_question.unsqueeze(dim=1).expand_as(unsorted_passage)
        
        # 4.2. Concatenate to make the combined representation.
        # Hint: Use torch.cat
    
        combined_x_question = torch.cat([unsorted_passage, tiled_encoded_question,
                                  unsorted_passage * tiled_encoded_question], dim=-1)

        #dropout
        dropout_combined_x_question = self.dropout(combined_x_question)

        # Part 5: Compute logits for answer start index.

        # 5.1. Apply the affine transformation, and edit the shape.
        
        start_logits = self.start_output_projection(dropout_combined_x_question).squeeze(-1)
        

        # 5.2. Replace the masked values so they have a very low score (-1e7).
        # This tensor is your start_logits.
        # Hint: allennlp.nn.util.replace_masked_values might be helpful.
    
        start_logits = replace_masked_values(start_logits, passage_mask, -1e7)

        # 5.3. Apply a padding-aware log-softmax to normalize.
        # This tensor is your softmax_start_logits.
        # Hint: allennlp.nn.util.masked_log_softmax might be helpful.
    
        softmax_start_logits = masked_log_softmax(start_logits, passage_mask)

        # Part 6: Compute logits for answer end index.

        # 6.1. Apply the affine transformation, and edit the shape.

        end_logits = self.end_output_projection(dropout_combined_x_question).squeeze(-1)
       
        # 6.2. Replace the masked values so they have a very low score (-1e7).
        # This tensor is your end_logits.
        # Hint: allennlp.nn.util.replace_masked_values might be helpful.
    
        end_logits = replace_masked_values(end_logits, passage_mask, -1e7)
       
        # 6.3. Apply a padding-aware log-softmax to normalize.
        # This tensor is your softmax_end_logits.
        # Hint: allennlp.nn.util.masked_log_softmax might be helpful.
    
        softmax_end_logits = masked_log_softmax(end_logits, passage_mask)

        # Part 7: Output a dictionary with the start_logits, end_logits,
        # softmax_start_logits, softmax_end_logits. Remove the NotImplementedError below.

        return {
            "start_logits": start_logits,
            "end_logits": end_logits,
            "softmax_start_logits": softmax_start_logits,
            "softmax_end_logits": softmax_end_logits,
        }
