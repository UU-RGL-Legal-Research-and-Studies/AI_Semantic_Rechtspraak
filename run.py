from app import app

if __name__ == '__main__':
    app.run(debug=True, port=5000)
    
#in WSL: sudo service redis-server start
#in WSL: redis-cli
