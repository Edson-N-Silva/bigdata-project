import bcrypt

# Use este script para gerar hashes seguros para suas senhas.
# Execute no terminal com: python generate_keys.py

def generate_hashes(passwords_list):
    """
    Gera hashes seguros para uma lista de senhas usando bcrypt.
    """
    hashed_passwords = []
    for password in passwords_list:
        # Codifica a senha para bytes, gera o salt e o hash
        password_bytes = password.encode('utf-8')
        salt = bcrypt.gensalt()
        hashed_password = bcrypt.hashpw(password_bytes, salt)
        # Decodifica o hash para string para salvar no arquivo .yaml
        hashed_passwords.append(hashed_password.decode('utf-8'))
    return hashed_passwords

if __name__ == '__main__':
    # --- COLOQUE AS SENHAS QUE VOCÊ QUER CRIPTOGRAFAR AQUI ---
    plain_text_passwords = ['123456789', '123123123', 'admin']
    
    print("Gerando hashes para as senhas...")
    
    hashed_passwords_list = generate_hashes(plain_text_passwords)
    
    print("\nHashes gerados com sucesso!")
    print("---------------------------------")
    print("Copie a lista abaixo e use no seu arquivo config.yaml:")
    
    # Imprime a lista de hashes para você copiar e colar
    print(f"\n{hashed_passwords_list}\n")

