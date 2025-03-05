python converter_aquaformat.py --input_directory "directory with jsons" --include_mod_utterances True
				               
Visit the following repo: https://github.com/mabehrendt/AQuA

Follow the instructions e.g., 

python translate_to_german.py "path to the csv file to be translated" "dataset name" "translation column (en_text)" "path to output file"

python inference_parallel_de.py "path to the translated csv file" "column with comment in german (comment_de)" "batch_size" "output_path"