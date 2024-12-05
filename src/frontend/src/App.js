import React, { useState, useEffect } from "react";
import "bootstrap/dist/css/bootstrap.min.css";

import Home from "./components/Home";
import Login from "./components/Login";
import Forgot from "./components/Forgot";
import UserManagement from "./components/UserManagement";
import renderMainMenu from "./components/Navbar";
import ChartComponent from './components/ProcessedDataChart';
import PreprocessData from './components/PreprocessData';
// import PreprocessorManagement from "./components/PreprocessorManagement";
// import ModelManagement from "./components/ModelManagement";
// import PreprocessingCompare from "./components/PreprocessingCompare";
// import PreprocessingEditConfigs from "./components/PreprocessingEditConfigs";
// import TrainingEditConfigs from "./components/TrainingEditConfigs";
// import TrainingPrediction from "./components/TrainingPredictions";
import "./styles/App.css";
// import PredictionsEvolution from "./components/PredictionsEvolution";

function App() {
    const [page, setPage] = useState("login");
    const [token, setToken] = useState(null);
    const [userRole, setUserRole] = useState(null);
    const [submenuSelection, setSubmenuSelection] = useState(null);

    useEffect(() => {
        const auth = localStorage.getItem("auth_token");
        const role = localStorage.getItem("user_role");
        setToken(auth);
        setUserRole(role);
    }, []);

    const chosePage = () => {
        switch (page) {
            case "login":
                return <Login setPage={setPage} />;
            case "forgot":
                return <Forgot setPage={setPage} />;
            default:
                return <Login setPage={setPage} />;
        }
    };

    const renderSubmenuSelection = () => {
        switch (submenuSelection) {
            case "usermanagement":
                return <UserManagement />;
            // case "preprocessors":
            //     return <PreprocessorManagement />;
            // case "models":
            //     return <ModelManagement />;
            // case "outliers":
            //     return <OutliersManagement />;
            case "preprocessing-visualize":
                return <ChartComponent />;
            case "preprocessing-data":
                return <PreprocessData />;
            // case "preprocessing-compare":
            //     return <PreprocessingCompare />;
            // case "preprocessing-edit":
            //     return <PreprocessingEditConfigs />;
            // case "training-edit":
            //     return <TrainingEditConfigs />;
            // case "training-predictions":
            //     return <TrainingPrediction />;
            // case "reports-predictions":
            //     return <PredictionsEvolution />;
            case "logout":
                localStorage.removeItem("auth_token");
                localStorage.removeItem("user_role");
                window.location.reload();
                break;
            default:
                return <Home />;
        }
    };

    const handleSubmenuSelection = (selection) => {
        setSubmenuSelection(selection);
    };

    const pages = () => {
        if (!token) {
            return (
                <div className="min-h-screen bg-yellow-400 flex justify-center items-center">
                    <div className="py-12 px-12 bg-white rounded-2xl shadow-xl z-20">
                        {chosePage()}
                    </div>
                </div>
            );
        } else {
            return (
                <div>
                    {renderMainMenu(handleSubmenuSelection)}
                    {renderSubmenuSelection()}
                </div>
            );
        }
    };

    return <React.Fragment>{pages()}</React.Fragment>;
}

export default App;