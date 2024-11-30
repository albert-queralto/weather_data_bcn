import React, { useState } from "react";
import { Link } from "react-router-dom";
import axios from "axios";
import { toast } from "react-toastify";

export default function Login(props) {
    const [loginForm, setLoginForm] = useState({
        email: "",
        password: "",
        created_date: new Date(),
        update_date: new Date(),
        role: "",
    });

    const onChangeForm = (label, event) => {
        switch (label) {
            case "email":
                setLoginForm({ ...loginForm, email: event.target.value });
                break;
            case "password":
                setLoginForm({ ...loginForm, password: event.target.value });
                break;
        }
    };

    const onSubmitHandler = async (event) => {
        event.preventDefault();
        await axios
            .post("http://localhost:8100/signin", loginForm)
            .then((response) => {
                localStorage.setItem("auth_token", response.data.token.access_token);
                localStorage.setItem(
                    "auth_token_type",
                    response.data.token.token_type
                );
                localStorage.setItem("user_role", response.data.role);
                
                toast.success(response.data.detail);

                setTimeout(() => {
                    window.location.reload();
                }, 1000);
            })
            .catch((error) => {
                console.log(error);
                toast.error(error.response.data.detail);
            });
    };



    return (
        <React.Fragment>
            <header className="chat-header">
                <div style={{ display: "flex", justifyContent: "flex-start", alignItems: "center", width: "100%" }}>
                    <div style={{ flexGrow: 1, display: "flex", justifyContent: "center" }}>
                        Model trainer application
                    </div>
                </div>
            </header>
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", width: "100%" }}>
                <div style={{ width: "100%", display: "flex", justifyContent: "center" }}>
                    <p className="w-80 text-center text-sm mb-8 font-semibold text-gray-700 tracking-wide cursor-pointer">
                        Please login to continue
                    </p>
                </div>
            </div>
            <form onSubmit={onSubmitHandler}>
                <div className="container d-flex flex-column">
                    <div className="row justify-content-center">
                        <div className="col-md-4 d-flex justify-content-center">
                            <input
                                type="email"
                                placeholder="Email"
                                className="px-4 py-3 text-sm border outline-none focus:ring focus:outline-none flex-grow rounded-pill"
                                onChange={(event) => {onChangeForm("email", event);}}
                            />
                        </div>
                        <div className="col-md-4 d-flex justify-content-center">
                            <input
                                type="password"
                                placeholder="Password"
                                className="px-4 py-3 text-sm border outline-none focus:ring focus:outline-none flex-grow rounded-pill"
                                onChange={(event) => {onChangeForm("password", event);}}
                            />
                        </div>
                        <div className="col-md-4 d-flex justify-content-center">
                            <button
                                type="submit"
                                className="px-4 py-3 text-xl text-white outline-none rounded-pill"
                            >
                                Login
                            </button>
                        </div>
                    </div>
                </div>
                <div className="text-center mt-6">
                    <p className="mt-4 text-sm">
                        <Link
                        to="/forgot"
                        onClick={() => {
                            props.setPage("forgot");
                        }}
                        >
                        <span className="underline cursor-pointer">Forgot Password?</span>
                        </Link>
                    </p>
                </div>
            </form>
        </React.Fragment>
    );
}