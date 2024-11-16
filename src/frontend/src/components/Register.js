import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import axios from "axios";
import { toast } from "react-toastify";
import 'react-toastify/dist/ReactToastify.css';

export default function Register(props) {
    const options = [
        { value: "", label: "Select the role" },
        { value: "admin", label: "Admin" },
        { value: "user", label: "User" },
    ];
    
    const navigate = useNavigate();

    const [registerForm, setRegisterForm] = useState({
        email: "",
        password: "",
        role: "",
        created_date: new Date(),
        update_date: new Date(),
    });

    const onChangeForm = (label, event) => {
        switch (label) {
            case "email":
                setRegisterForm({ ...registerForm, email: event.target.value });
                break;
            case "password":
                setRegisterForm({ ...registerForm, password: event.target.value });
                break;
            case "role":
                setRegisterForm({ ...registerForm, role: event.target.value });
                break;
        }
    };

    const onSubmitHandler = async (event) => {
        event.preventDefault();
        await axios
            .post("/api/signup", registerForm)
            .then((response) => {
                navigate("/signin");

                toast.success(response.data.detail);

                setTimeout(() => {
                    window.location.reload();
                }, 1000);

                console.log(response);
            })
            .catch((error) => {
                console.log(error);
                toast.error(error.response.data.detail);
            });
    };

    return (
        <React.Fragment>
            {/* <div>
                <header className="chat-header">
                    <div style={{ display: "flex", justifyContent: "flex-start", alignItems: "center", width: "100%" }}>
                        <div style={{ flexGrow: 1, display: "flex", justifyContent: "center" }}>
                            Create An Account
                        </div>
                    </div>
                </header>
                <p className="w-80 text-center text-sm mb-8 font-semibold text-gray-700 tracking-wide cursor-pointer mx-auto">
                    Welcome to the Chat App. Please create an account to continue.
                </p>
            </div> */}
            <form onSubmit={onSubmitHandler}>
                <div className="container d-flex flex-column">
                    <div className="row justify-content-center">
                        <div className="col-md-3 d-flex justify-content-center">
                            <input
                                type="email"
                                placeholder="Email"
                                className="block text-sm py-3 px-4 w-full border outline-none focus:ring focus:outline-none rounded-pill"
                                onChange={(event) => {onChangeForm("email", event);}}
                            />
                        </div>
                        <div className="col-md-3 d-flex justify-content-center">
                            <input
                                type="password"
                                placeholder="Password"
                                className="block text-sm py-3 px-4 w-full border outline-none focus:ring focus:outline-none rounded-pill"
                                onChange={(event) => {onChangeForm("password", event);}}
                            />
                        </div>
                        <div className="col-md-3 d-flex justify-content-center">
                            <select
                                value={registerForm.role}
                                className="block text-sm py-3 px-4 w-full border outline-none focus:ring focus:outline-none rounded-pill"
                                onChange={(event) => {onChangeForm("role", event);}}
                            >
                                {options.map((data) => {
                                    if (data.value === "") {
                                        return (
                                        <option key={data.label} value={data.value} disabled>
                                            {data.label}
                                        </option>
                                        );
                                    } else {
                                        return (
                                        <option key={data.label} value={data.value}>
                                            {data.label}
                                        </option>
                                        );
                                    }
                                })}
                            </select>
                        </div>
                        <div className="col-md-3 d-flex justify-content-center">
                            <button
                                type="submit"
                                className="py-3 w-64 text-xl text-white rounded-pill outline-none"
                            >
                                Create Account
                            </button>
                        </div>
                    </div>
                </div>
                <div className="text-center mt-6">
                    <p className="mt-4 text-sm">
                        Already have an account?{" "}
                        <Link
                        to="/signin"
                        onClick={() => {
                            props.setPage("login");
                        }}
                        >
                        <span className="underline cursor-pointer">Sign In</span>
                        </Link>
                    </p>
                </div>
            </form>
            <ToastContainer />
        </React.Fragment>
    );
}