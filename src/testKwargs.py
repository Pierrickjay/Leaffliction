# mon_script.py
import argparse

def main(**kwargs):
    print("Arguments reçus :", kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exemple de passage d'arguments de mots-clés.")
    
    # Ajoutez les arguments de mots-clés nécessaires
    parser.add_argument("--argument1", type=int, help="Description de l'argument 1")
    parser.add_argument("--argument2", type=str, help="Description de l'argument 2")
    
    # Analysez les arguments de la ligne de commande
    args = parser.parse_args()
    
    # Appelez la fonction principale avec les arguments de mots-clés
    main(argument1=args.argument1, argument2=args.argument2)
