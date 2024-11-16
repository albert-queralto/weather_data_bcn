import React from 'react';

const Home = () => {
    return (
        <div>
            <header className="chat-header">
                <div style={{ flexGrow: 1, display: "flex", justifyContent: "center" }}>
                    Welcome to the Home Page
                </div>
            </header>
            <main>
                <p>This is the home page of our React application.</p>
            </main>
        </div>
    );
};

export default Home;