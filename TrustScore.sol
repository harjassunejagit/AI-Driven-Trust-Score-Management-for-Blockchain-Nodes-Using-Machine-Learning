// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title TrustScore
/// @notice On-chain immutable storage for AI-computed blockchain-node trust
///         scores. Reconstructed to match TrustScore_abi.json exactly (the
///         event, updateTrust, getTrust and trustScores signatures) and the
///         Remix compiler settings used to originally deploy it: solc
///         0.8.20, optimizer enabled with 200 runs, EVM version istanbul.
///         The original source file was misplaced after deployment — this
///         is a faithful reconstruction from the ABI and the paper's
///         described access-control behavior (Section 3.9 / Algorithm 5),
///         not a byte-for-byte recovery of the original file. Verify against
///         the deployed bytecode if you still have it, before citing this
///         as the canonical source.
contract TrustScore {
    /// @notice The single authorised off-chain oracle account (the Python
    ///         trust evaluation engine's sender address) allowed to write
    ///         trust scores. Set once at deployment to the deploying account.
    address public owner;

    /// @notice node address => current trust score (0-100 in this project's
    ///         usage, though the type itself does not enforce that range).
    mapping(address => uint256) public trustScores;

    event TrustUpdated(address indexed node, uint256 newScore);

    modifier onlyOwner() {
        require(msg.sender == owner, "TrustScore: caller is not the authorised oracle");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    /// @notice Write a new trust score for `node`. Reverts if called by
    ///         anyone other than the authorised oracle account.
    function updateTrust(address node, uint256 newScore) external onlyOwner {
        trustScores[node] = newScore;
        emit TrustUpdated(node, newScore);
    }

    /// @notice Read the current trust score for `node`. Callable by anyone,
    ///         free of gas cost off-chain (view function).
    function getTrust(address node) external view returns (uint256) {
        return trustScores[node];
    }
}
