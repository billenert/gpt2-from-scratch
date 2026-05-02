from dataset import FineWebEduStream
import tiktoken
test = FineWebEduStream(batch_size=1, seq_len=100)
batches = iter(test)
tokenizer = tiktoken.get_encoding("gpt2")      
counter = 0
for batch in batches:
    counter += 1
    if counter == 100: 
        break
    for row in batch:                                                                                                                                                                                                                          
        print(tokenizer.decode(row.tolist()))
