using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntimeGenAI;


namespace HyperGenAI {
	public class HGInferSession {

		private readonly Model _model;
		private readonly Tokenizer _tokenizer;

		public HGInferSession(string modelpath) {
			_model = new Model(modelpath);
			_tokenizer = new Tokenizer(_model);


		}


		public List<string> Infer(string input, int top_k=50, float top_p=0.9f, float temp=0.75f, float repeatPenalty=1f, int maxLength=100, bool doSample=true) {
			List<string> outputs = new List<string>();
			var parameters = GetGeneratorParams(top_k, top_p, temp,repeatPenalty,maxLength,doSample);
			var tokens = _tokenizer.Encode(input);
			parameters.SetInputSequences(tokens);
			var tempGenerator = new Generator(_model,parameters);
			while(!tempGenerator.IsDone()) {
				tempGenerator.ComputeLogits();
				tempGenerator.GenerateNextToken();
				var otokens = tempGenerator.GetSequence(0);
				var lasttoken = otokens.Slice(otokens.Length-1,1);
				var lasttokendecoded = _tokenizer.Decode(lasttoken);
				outputs.Add(lasttokendecoded);
			}
			return outputs;
		}

		private GeneratorParams GetGeneratorParams(int top_k, float top_p, float temp, float repeatPenalty, int maxLength, bool doSample) {
			var parameters = new GeneratorParams(_model);
			parameters.SetSearchOption("top_k", 50);
			parameters.SetSearchOption("top_p", 0.9f);
			parameters.SetSearchOption("temperature", 0.75f);
			parameters.SetSearchOption("repetition_penalty", 1f);
			parameters.SetSearchOption("max_length", maxLength);
			parameters.SetSearchOption("do_sample", doSample);
			//sane defaults for the rest
			//TODO: Expose these as parameters
			parameters.SetSearchOption("num_return_sequences", 1);
			parameters.SetSearchOption("num_beams", 1);
			parameters.SetSearchOption("no_repeat_ngram_size", 0);
			parameters.SetSearchOption("early_stopping", false);
			parameters.SetSearchOption("length_penalty", 1f);
			parameters.SetSearchOption("num_beam_groups", 1);
			parameters.SetSearchOption("diversity_penalty", 0f);
			
			return parameters;
		}
	}
}
