import React, { useState } from "react";
import { Link } from "react-router-dom";
import axios from "axios";
import { toast } from "react-toastify";

export default function Forgot(props) {
  const [forgotForm, setForgotForm] = useState({
    email: "",
    new_password: "",
  });

  const onChangeForm = (label, event) => {
    switch (label) {
      case "email":
        setForgotForm({ ...forgotForm, email: event.target.value });
        break;
      case "new_password":
        setForgotForm({ ...forgotForm, new_password: event.target.value });
        break;
    }
  };

  const onSubmitHandler = async (event) => {
    event.preventDefault();
    const { email, new_password } = forgotForm;
    const user = {
      email: email,
      password: new_password,
      role: "",
      created_date: new Date(),
      update_date: new Date(),
    };
  
    try {
      const response = await axios.put(`/api/users/${email}`, user);
      toast.success(response.data.message);
      setTimeout(() => {
        window.location.reload();
      }, 1000);
    } catch (error) {
      toast.error(error.response.data.detail);
    }
  };

  return (
    <React.Fragment>
      <div>
        <header className="chat-header">
            <div style={{ display: "flex", justifyContent: "flex-start", alignItems: "center", width: "100%" }}>
                <div style={{ flexGrow: 1, display: "flex", justifyContent: "center" }}>
                    Forgot your password?
                </div>
            </div>
        </header>
        <p className="w-80 text-center text-sm mb-8 font-semibold text-gray-700 tracking-wide cursor-pointer mx-auto">
          Now update your password account!
        </p>
      </div>
      <form onSubmit={onSubmitHandler}>
          <div className="container d-flex flex-column">
            <div className="row justify-content-center">
              <div className="col-md-4 d-flex justify-content-center">
                <input
                  type="email"
                  placeholder="Email"
                  className="block text-sm py-3 px-4 rounded-lg w-full border outline-none focus:ring focus:outline-none rounded-pill"
                  onChange={(event) => {
                    onChangeForm("email", event);
                  }}
                />
              </div>
              <div className="col-md-4 d-flex justify-content-center">
                <input
                  type="password"
                  placeholder="New Password"
                  className="block text-sm py-3 px-4 rounded-lg w-full border outline-none focus:ring focus:outline-none rounded-pill"
                  onChange={(event) => {
                    onChangeForm("new_password", event);
                  }}
                />
              </div>
              <div className="col-md-4 d-flex justify-content-center">
                <button
                  type="submit"
                  className="py-3 w-64 text-xl text-white rounded-2xl outline-none rounded-pill"
                >
                  Update Password
                </button>
              </div>
            </div>
          </div>
        <div className="text-center mt-6">
          <p className="mt-4 text-sm">
            Remember your password?{" "}
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
    </React.Fragment>
  );
}