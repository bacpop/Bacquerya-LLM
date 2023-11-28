from lib.llm import start_language_models


def summarise_articles(args):
    import json
    from lib.summariser import Summariser
    summariser_model = start_language_models(
        lang_model_path=args.language_model,
        emb_model_path=args.embedding_model,
        device_map=args.device_map,
        model_type=args.model_type,
        chat_type=args.chat_type,
        n_ctx=args.n_ctx
    )

    summariser = Summariser(summariser_model, texts=args.articles_json)
    summarised_articles = summariser.process()
    with open(args.output_file, "w") as f:
        json.dump(summarised_articles, f)


if __name__ == "__main__":
    import argparse
    argparse = argparse.ArgumentParser()

    argparse.add_argument("--language_model", type=str,required=True, help="Main LLM model for summarisation")
    argparse.add_argument("--embedding_model", type=str, default=None, required=False, help="Sentence embedding model to calculate similarity")
    argparse.add_argument("--device_map", type=str, default='cuda', required=False, help="Device map for LLM model")
    argparse.add_argument("--model_type", type=str, default=None, required=False, help="LLM model type")
    argparse.add_argument("--chat_type", type=str, default=None, required=False, help="LLM chat type")
    argparse.add_argument("--n_ctx", type=int, default=8000, required=False, help="Number of context for LLM model")
    argparse.add_argument("--articles_json", type=str, required=True, help="JSON file containing articles")
    argparse.add_argument("--output_file", type=str, required=True, help="Output file containing summarised articles")

    args = argparse.parse_args()
    summarise_articles(args)


