namespace Example {
	internal class Program {
		static void Main(string[] args) {
			Console.WriteLine("Loading Model");
			var session = new HyperGenAI.HGInferSession(@"D:\models\onnx\Phi-3-mini-4k-instruct-onnx");
			while(true) {
				Console.WriteLine("Enter input: ");
				var input = Console.ReadLine();
				var outputs = session.Infer(input);
				foreach(var output in outputs) {
					Console.Write(output);
				}
				Console.WriteLine("---");
			}

		}
	}
}
